# -*- coding: utf-8 -*-
import os
import sys
import argparse


import numpy as np

import faster2lib.eeg_tools as et
import faster2lib.timeseries_graph as tg

from datetime import datetime

import multiprocessing

from glob import glob
import re
import json
import shutil
import zipfile

def get_day_idx(file_path, epoch_len_sec, pat):
    # file_path: a string of the file path
    # pat: compiled regex pattern
    matched = pat.search(file_path)
    epoch_idx_str = matched.group(1)
    epoch_idx = int(epoch_idx_str)
    day_idx = int(np.floor(epoch_idx * epoch_len_sec / 86400) + 1)  
    
    return day_idx

def make_archive(result_dir, epoch_num, epoch_len_sec, device_id):
    """ make archives of the plots

    Args:
        result_dir (str): The path to the folder containing the figure/voltage
        epoch_num (int): The total epoch number in the measurements
        epoch_len_sec (int): The length of an epoch in seconds
        device_id (str): The device ID
    """
    # file list of deviceID's plots
    file_list = glob(os.path.join(result_dir, 'figure', 'voltage', f'{device_id}_tmp', '*.jpg'))
    file_list.sort()

    # make a dictionary of plots' list for each day
    pat = re.compile(r'(\d+)\.jpg$')
    zip_files = {}
    for t_file in file_list:
        day_idx = get_day_idx(t_file, epoch_len_sec, pat)
        if day_idx in zip_files:
            zip_files[day_idx].append(t_file)
        else:
            zip_files[day_idx] = [t_file]

    # create the zip file
    plot_dir = os.path.join(result_dir, 'figure', 'voltage', f'{device_id}')
    os.makedirs(plot_dir, exist_ok=True)
    for day_idx, target_files in zip_files.items():
        zipped_file = os.path.join(plot_dir, f'{device_id}_day{day_idx:02}.zip')
        print(f'Making zip file: {zipped_file}')
        with zipfile.ZipFile(zipped_file, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for t_file in target_files:
                zf.write(t_file, os.path.basename(t_file))

    # create a json file for signal_view
    signal_view_info = {"mes_len_sec": epoch_num * epoch_len_sec,
        "epoch_len_sec": epoch_len_sec,
        "epoch_per_page": int(tg.row_len_sec(epoch_len_sec)*5/epoch_len_sec)
     }
    with open(os.path.join(plot_dir, 'signal_view.json'), 'w') as f_json:
        json.dump(signal_view_info, f_json, indent=4)

    # cleaning the temporary directory
    shutil.rmtree(os.path.join(result_dir, 'figure', 'voltage', f'{device_id}_tmp'))


def main(args):
    """The main function

    Args:
        args (PARSER.args): Arguments given at the command line
    """
    epoch_len_sec = int(args.epoch_len_sec)

    data_dir = os.path.abspath(args.data_dir)
    exp_info_df = et.read_exp_info(data_dir)
    (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime) = et.interpret_exp_info(exp_info_df, epoch_len_sec)

    stage_dir = os.path.join(args.result_dir)

    mouse_info_df = et.read_mouse_info(data_dir)

    # The epoch length must be a divisor of the length of a day
    if (86400 % epoch_len_sec) != 0:
        print(f'[Error] The given epoch length: {epoch_len_sec} is not a divisor of 86400 i.e. seconds of a day.')
        sys.exit(1)

    dt_now = datetime.now()
    print(f'Started plotting voltage time-series: {dt_now}')
    if args.workers is None:
        # draw timeseries plots mouse by mouse
        for i, r in mouse_info_df.iterrows():
            device_id = r.iloc[0]

            tg.plot_timeseries_a_mouse(data_dir, stage_dir, stage_dir, device_id, sample_freq, epoch_num, epoch_len_sec, start_datetime)
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
            pss = [multiprocessing.Process(target=tg.plot_timeseries_a_mouse, args=(
                data_dir, stage_dir, stage_dir, device_ids[i], sample_freq, epoch_num, epoch_len_sec, start_datetime)) for i in range(len(device_ids)) if device_ids[i] != None]
            # start them
            for ps in pss:
                ps.start()
            # wait for all finish
            for ps in pss:
                ps.join()
    
    # zip the plots
    for device_id in mouse_info_df['Device label'].values:
        make_archive(stage_dir, epoch_num, epoch_len_sec, device_id)


    elapsed_time = (datetime.now() - dt_now)
    print(f'Ended plotting: {datetime.now()},  ellapsed {elapsed_time.total_seconds()/60} minuites')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-d", "--data_dir", required=True, help="path to the directory of input voltage data")
    PARSER.add_argument("-r", "--result_dir", required=True, help="path to the directory of the input stage data and plots to be produced")
    PARSER.add_argument("-w", "--workers", type=int, help="number of worker processes to draw in parallel")
    PARSER.add_argument("-l", "--epoch_len_sec", help="epoch length in second", default=8)

    ARGS = PARSER.parse_args()
    main(ARGS)