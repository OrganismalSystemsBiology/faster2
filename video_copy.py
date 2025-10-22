# -*- coding: utf-8 -*-
from glob import glob
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter
import logging
import os
import shutil
import json
import argparse
import numpy as np
import traceback


APP_NAME = 'video_copy.py ver 0.0.0'

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
    parser.add_argument("-p", "--profile", required=True, help="parameter json file")
    parser.add_argument("-d", "--dest_dir", required=True, help="destination directory")
    parser.add_argument("-s", "--start", required=True, help="start datetime '2021-02-16 08:00:00'")
    parser.add_argument("-e", "--end", required=True, help="end datetime '2021-02-16 08:00:00'")

    args = parser.parse_args()
    profile_filepath = os.path.abspath(args.profile)
    dest_dir = os.path.abspath(args.dest_dir)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_et = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")

    os.makedirs(dest_dir, exist_ok=True)

    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(dest_dir, f'video_copy.{dt_str}.log'))

    profile_dict = load_profile(profile_filepath)

    main(profile_dict, dest_dir, start_dt, end_et)
