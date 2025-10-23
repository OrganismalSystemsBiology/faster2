# -*- coding: utf-8 -*-
from glob import glob
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter
import logging
import os
import shutil
import json
import argparse
import traceback
import numpy as np

import faster2lib.eeg_tools  as et


APP_NAME = 'video_copy.py ver 0.1.0'

def initialize_logger(log_file):
    """initializes the log

    Args:
        log_file (str): The filepath to the log
    """
    logger = getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)

    file_handler = FileHandler(log_file)
    stream_handler = StreamHandler()

    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.NOTSET)
    handler_formatter = Formatter('%(message)s')
    file_handler.setFormatter(handler_formatter)
    stream_handler.setFormatter(handler_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def print_log(msg):
    """ print the given message to the log if defined or to the stdout

    Args:
        msg (str): A string to print
    """
    if 'log' in globals():
        log.debug(msg)
    else:
        print(msg)


def load_profile(path):
    """ It reads a profile json file.

    Args:
        path (str): the filepath to the profile json file

    Returns:
        dict: the dict of the profile
    """
    with open(path, 'r') as fhandle:
        profile = json.load(fhandle)

    return profile


def time_from_filepaths(file_list, timefmt):
    """ It returns a list of datetime by interpreting filepaths
    according to the given timefmt

    Args:
        file_list ([str]): a list of filepaths

    Returns:
        [datetime]: a list of datetime
    """
    time_list=[]
    try:
        time_list = np.sort(np.array(
            [datetime.strptime(
                os.path.splitext(os.path.basename(file_path))[0],
                timefmt) for file_path in file_list]
        ))
    except ValueError as val_err:
        print(traceback.format_exc())
        print('The following filelist contains irregular video files')
        print(file_list)
        raise ValueError('Irregular video file found in the list') from val_err
    
    return time_list


def show_parameters(faster2_dir, profile_path, dest_dir, 
                    start_dt, end_dt, exp_label, rack_label):
    """Display key parameters before executing main.

    Args:
        faster2_dir (str): Absolute path to the faster2 directory
        profile_path (str): Path to the profile JSON used
        dest_dir (str): Destination directory for copied videos
        tmp_dir (str): Temporary/log output directory from profile
        start_dt (datetime): Start datetime for copy range
        end_dt (datetime): End datetime for copy range
        exp_label (str): Experiment label from exp.info.csv
        rack_label (str): Rack label from exp.info.csv
    """
    print_log("Parameters")
    print_log(f"  faster2_dir : {faster2_dir}")
    print_log(f"  profile_path: {profile_path}")
    print_log(f"  dest_dir    : {dest_dir}")
    print_log(f"  start_dt    : {start_dt}")
    print_log(f"  end_dt      : {end_dt}")
    print_log("exp.info.csv")
    print_log(f"  exp_label   : {exp_label}")
    print_log(f"  rack_label  : {rack_label}")


def main(profile, dest_root_dir, copy_start, copy_end):
    source_dir_list = profile['source_dir_list']
    timefmt_list = profile['timefmt_list']
    ext_list = profile['ext_list']
    dest_camera_list = profile['dest_camera_list']
    dest_timefmt = profile['dest_timefmt']

    for source_dir, timefmt, ext, dest_camera in zip(source_dir_list, timefmt_list, ext_list, dest_camera_list):
        file_list = np.array(
            glob(os.path.join(source_dir, f'**/*.{ext}'), recursive=True))
        try:
            time_list = time_from_filepaths(file_list, timefmt)
        except ValueError:
            exit(-1)

        dest_camera_dir = os.path.join(dest_root_dir, dest_camera)

        print_log(f'Source: {source_dir}')
        print_log(f'Destination: {dest_camera_dir}')
        os.makedirs(dest_camera_dir, exist_ok=True)

        # make the index list of the copy targets
        bidx_source = (time_list > copy_start) & (time_list < copy_end)
        idx_edges = np.where(bidx_source[1:] ^ bidx_source[:-1])[0]
        if len(idx_edges) == 2:
            # the target is within the source
            idx_s = idx_edges[0]
        elif len(idx_edges) == 1 and bidx_source[0]:
            # the target starts before the source
            idx_s = 0
        elif len(idx_edges) == 1 and bidx_source[-1]:
            # the target ends after the source
            idx_s = idx_edges[0]
        else:
            print_log('No file found in the period')
            continue

        bidx_source[idx_s] = True  # include one previous video to the source

        num_all = np.sum(bidx_source)
        num_cur = 0
        # pylint: disable=invalid-name
        for s_path, dt in zip(file_list[bidx_source], time_list[bidx_source]):
            num_cur += 1
            s_fn = os.path.basename(s_path)
            d_fn = f'{dt.strftime(dest_timefmt)}.{ext}'
            d_path = os.path.join(dest_camera_dir, d_fn)
            shutil.copy2(s_path, d_path)

            print_log(f'Copying [{num_cur}/{num_all}] {s_fn} to {d_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--faster2_dir", required=True, help="faster2 directory")
    parser.add_argument("-p", "--profile", required=False, help="profile name (video.[HERE].json)")
    parser.add_argument("-d", "--dest_dir", required=False, help="destination directory")
    parser.add_argument("-s", "--start", required=False, help="start datetime '2021-02-16 08:00:00'")
    parser.add_argument("-e", "--end", required=False, help="end datetime '2021-02-16 08:00:00'")


    # Parameters (1/3): command line
    args = parser.parse_args()
    faster2_dir = os.path.abspath(args.faster2_dir)

    # Parametes (2/3): exp_info csv
    exp_info_df = et.read_exp_info(os.path.join(faster2_dir, 'data'))
    exp_label = exp_info_df["Experiment label"].iloc[0]
    rack_label = exp_info_df["Rack label"].iloc[0]


    # Set parameters depending on their priority ( command line > exp_info ) (1/2)
    if args.profile is None:
        profile_path = os.path.abspath(os.path.join(faster2_dir, f'video.{rack_label}.json'))
    else:
        profile_path = os.path.abspath(os.path.join(faster2_dir, f'video.{args.profile}.json'))

    if args.start is None:
        start_dt = et.interpret_datetimestr(exp_info_df["Start datetime"].iloc[0])
    else:
        start_dt = et.interpret_datetimestr(args.start)

    if args.end is None:
        end_dt = et.interpret_datetimestr(exp_info_df["End datetime"].iloc[0])
    else:
        end_dt = et.interpret_datetimestr(args.end)

    # Parameters (3/3): profile json
    try:
        profile_dict = load_profile(profile_path)
    except FileNotFoundError as e:
        print_log(f"Error: Profile file not found: {profile_path}")
        print_log("Please ensure:")
        print_log("  1. The 'Rack label' in exp.info.csv is correct, OR")
        print_log("  2. Use --profile option to specify the correct profile name")
        print_log("     (e.g., --profile rack1 for video.rack1.json)")
        print_log("  3. The specified json file is in the faster2_dir")
        exit(-1)
    temp_dir = os.path.abspath(os.path.join(profile_dict['tmp_dir'], os.path.basename(faster2_dir)))

    os.makedirs(temp_dir, exist_ok=True)

    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(temp_dir, f'video_copy.{dt_str}.log'))

    show_parameters(
        faster2_dir=faster2_dir,
        profile_path=profile_path,
        dest_dir=temp_dir,
        start_dt=start_dt,
        end_dt=end_dt,
        exp_label=exp_label,
        rack_label=rack_label,
    )

    main(profile_dict, temp_dir, start_dt, end_dt)
