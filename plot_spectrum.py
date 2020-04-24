# -*- coding: utf-8 -*-
import os
import argparse


import numpy as np
import pandas as pd

import stage
import spectrum_graph as sg

import concurrent.futures
from datetime import datetime

import multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, help="path to the directory of input voltage data")
    parser.add_argument("-r", "--result_dir", required=True, help="path to the directory of the input stage data and plots to be produced")
    parser.add_argument("-w", "--workers", type=int, help="number of worker processes to draw in parallel")

    args = parser.parse_args()

    data_dir = args.data_dir
    result_dir = args.result_dir
    psd_data_dir = os.path.join(result_dir, 'PSD')
    cluster_params_dir = os.path.join(result_dir, 'cluster_params')

    exp_info_df = stage.read_exp_info(data_dir)
    (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime) = stage.interpret_exp_info(exp_info_df)

    mouse_info_df = stage.read_mouse_info(data_dir)


    dt_now = datetime.now()
    print(f'started plotting: {dt_now}')
    if args.workers == None:
        # draw timeseries plots mouse by mouse
        for i, r in mouse_info_df.iterrows():
            device_id = r[0]
            sg.plot_specs_a_mouse(psd_data_dir, cluster_params_dir, result_dir, device_id, sample_freq)
    else:
        # draw timesereis plots in parallel
        w = args.workers

        device_ids = mouse_info_df['Device label'].values
        num_task = len(device_ids)
        min_multi = int(w * np.ceil(num_task / w))
        device_ids_mat = np.concatenate(
            [device_ids, [None]*(min_multi-num_task)]).reshape(-1, w)

        for device_ids in device_ids_mat:
            # prepare w processes
            pss = [multiprocessing.Process(target=sg.plot_specs_a_mouse, args=(
                psd_data_dir, cluster_params_dir, result_dir, device_ids[i], sample_freq)) for i in range(len(device_ids)) if device_ids[i] != None]
            # start them
            for ps in pss:
                ps.start()
            # wait for all finish
            for ps in pss:
                ps.join()
    
    elapsed_time = (datetime.now() - dt_now)
    print(f'ended plotting: {dt_now},  ellapsed {elapsed_time.total_seconds()/60} minuites')
