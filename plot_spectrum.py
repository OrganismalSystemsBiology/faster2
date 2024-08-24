# -*- coding: utf-8 -*-
import os
import argparse


import numpy as np

import stage
import faster2lib.spectrum_graph as sg
import faster2lib.eeg_tools as et

from datetime import datetime

import multiprocessing

from glob import glob
import shutil
import zipfile

def make_archive(result_dir, device_id):
    """ make archives of the plots

    Args:
        result_dir (str): The path to the folder containing the figure/voltage
        epoch_num (int): The total epoch number in the measurements
        epoch_len_sec (int): The length of an epoch in seconds
        device_id (str): The device ID
    """
    # file list of deviceID's plots
    plot_dir = os.path.join(result_dir, 'figure', 'spectrum', f'{device_id}')
    folder_list = glob(os.path.join(plot_dir, '**/')) # only folders
    folder_list.sort()

    # create the zip file
    for t_folder in folder_list:
        file_list = glob(os.path.join(t_folder, '*.jpg'))
        zipped_file = os.path.dirname(t_folder) + '.zip'
        print(f'Making zip file: {zipped_file}')
        with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_STORED) as zip_fh:
            for t_file in file_list:
                zip_fh.write(t_file, os.path.basename(t_file))
        if (os.path.exists(zipped_file)):
            # remove the folder when it is successfully zipped
            shutil.rmtree(t_folder)


def main(args):
    """The main function

    Args:
        args (PARSER.args): The arguments given at the command line
    """

    data_dir = os.path.abspath(args.data_dir)
    result_dir = os.path.abspath(args.result_dir)
    psd_data_dir = os.path.join(result_dir, 'PSD')
    cluster_params_dir = os.path.join(result_dir, 'cluster_params')
    epoch_len_sec = int(args.epoch_len_sec)

    exp_info_df = et.read_exp_info(data_dir)
    (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime) = et.interpret_exp_info(exp_info_df, epoch_len_sec)

    mouse_info_df = et.read_mouse_info(data_dir)


    dt_now = datetime.now()
    print(f'Started plotting spectrums: {dt_now}')
    if args.workers is None:
        # draw timeseries plots mouse by mouse
        for i, r in mouse_info_df.iterrows():
            device_id = r[0]
            sg.plot_specs_a_mouse(psd_data_dir, cluster_params_dir, result_dir, device_id, sample_freq, epoch_num)
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
                psd_data_dir, cluster_params_dir, result_dir, device_ids[i], sample_freq, epoch_num)) for i in range(len(device_ids)) if device_ids[i] is not None]
            # start them
            for ps in pss:
                ps.start()
            # wait for all finish
            for ps in pss:
                ps.join()
    
    # zip the plots
    for device_id in mouse_info_df['Device label'].values:
        make_archive(result_dir, device_id)

    elapsed_time = (datetime.now() - dt_now)
    print(f'Ended plotting: {dt_now},  ellapsed {elapsed_time.total_seconds()/60} minuites')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-d", "--data_dir", required=True, help="path to the directory of input voltage data")
    PARSER.add_argument("-r", "--result_dir", required=True, help="path to the directory of the input stage data and plots to be produced")
    PARSER.add_argument("-w", "--workers", type=int, help="number of worker processes to draw in parallel")
    PARSER.add_argument("-l", "--epoch_len_sec", help="epoch length in second", default=8)

    ARGS = PARSER.parse_args()
    main(ARGS)
