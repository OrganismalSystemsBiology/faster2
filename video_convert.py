# -*- coding: utf-8 -*-
import argparse
import subprocess
from datetime import datetime
import os
from glob import glob
from logging import getLogger, StreamHandler, FileHandler, Formatter
import logging
import re
import json
import faster2lib.eeg_tools as et

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

def call_proc(input_filepath, output_dir, camera_id, video_start_dt_str, encoder):
    """ invokes an ffmpeg subprocess

    Arguments:
        input_filepath {str} -- a path to the video file to be converted
        output_dir {str} -- a path to the output directry
        camera_id {str} -- a label to be added in the resulting video file "[camera_id]_[video_start_dt].mp4"
        video_start_dt_str {str} -- a datetime string of the video start
        encoder {str} -- a string to be passed to ffmpeg's -c:v option for encoder (e.g. "h264", "libx264", or "h264_nvenc")

    Returns:
        dict -- "proc": subprocess object, "filestem": output filename without the extention 
    """
    output_filestem = f'{camera_id}_{video_start_dt_str}'
    output_filepath = os.path.join(
        output_dir, output_filestem + '.mp4')
    cmd = ['ffmpeg', '-i', input_filepath, '-c:v', encoder, '-an',
            '-b:v', '0', '-cq', '32', output_filepath, '-y']
    print(' '.join(cmd))

    # open a log file
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    with open(os.path.join(output_dir, 'logs', output_filestem + '.log'), 'w', encoding='utf-8') as logout:
        # start a subprocess
        proc = subprocess.Popen(cmd, stdout=logout, stderr=logout, text=True)
    
    return {'proc':proc, 'filestem':output_filestem}


def get_start_dt(video_filepath):
    """ obtain the timestamp of the video start

    Arguments:
        video_filepath {str} -- the path to the target video file

    Returns:
        datetime --- a datetime object of the start time
    """

    try:
        filename = os.path.basename(video_filepath)
        start_datetime = et.interpret_datetimestr(filename)
    except ValueError:
        print(f'[warning] Failed to interpret the start datetime from the filename "{filename}"')
        start_datetime = None

    return start_datetime


def load_profile(path):
    """ It reads a profile json file.

    Args:
        path (str): the filepath to the profile json file

    Returns:
        dict: the dict of the profile
    """
    with open(path, 'r', encoding='utf-8') as fhandle:
        profile = json.load(fhandle)

    return profile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--faster2_dir", required=False, help="path to the faster2 directory") 
    parser.add_argument("-t", "--target_dir", required=False, help="path to the target directory")
    parser.add_argument("-o", "--output_dir", required=False, help="path to the directory of the resulting video clips")
    parser.add_argument("-w", "--worker", help="the number of parallel workers", type=int, default=1)
    parser.add_argument("-e", "--encoder", help="a string to be passed to ffmpeg -c:v option for the encoder", default="libx264")

    args = parser.parse_args()
    worker_num = args.worker
    encoder = args.encoder

    args = parser.parse_args()


    # Set parameters depending on their priority ( command line > exp_info ) (1/2)
    if args.faster2_dir is None:
        try:
            target_dir = os.path.abspath(args.target_dir)
            output_dir = os.path.abspath(args.output_dir)
        except TypeError:
            print_log("Error: Please specify either --faster2_dir or both --target_dir and --output_dir")
            parser.print_help()
            exit(-1)
    else:
        # read the rack profile if faster2_dir is given
        faster2_dir = os.path.abspath(args.faster2_dir)
        try:
            # get rack label from exp.info.csv
            exp_info_df = et.read_exp_info(os.path.join(faster2_dir, 'data'))
            exp_label = exp_info_df["Experiment label"].iloc[0]
            rack_label = exp_info_df["Rack label"].iloc[0]

            profile_path = os.path.abspath(os.path.join(faster2_dir, f'video.{rack_label}.json'))
            profile_dict = load_profile(profile_path)
        except FileNotFoundError as e:
            print_log(f"Error: Profile file not found: {profile_path}")
            print_log("Please ensure:")
            print_log("  1. The 'Rack label' in exp.info.csv is correct, OR")
            print_log("  2. Use --target option to specify the target directory")
            print_log("  3. The specified json file is in the faster2_dir")
            exit(-1)

        target_dir = os.path.abspath(os.path.join(profile_dict['tmp_dir'], os.path.basename(faster2_dir)))
        output_dir = os.path.abspath(os.path.join(faster2_dir, 'video'))

    # recursively get contents of the target dir
    # assuming the structure of [camera_ids]/[video files]
    file_list = [f for f in glob(os.path.join(target_dir, '**/*')) if re.search(r'.*\.(avi|mp4)', f)]

    os.makedirs(target_dir, exist_ok=True)

    # open a log file
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(output_dir, f'video_copy.{dt_str}.log'))

    dt_now = datetime.now()
    print(f'started converting: {dt_now}')

    video_count = 0
    more_video = True
    processes =[]
    while more_video:   
        for i in range(worker_num):
            i_video_path = file_list[video_count]

            # get the parent dir name as a camera id
            camera_dir_path, video_filename = os.path.split(i_video_path)
            camera_id = os.path.split(camera_dir_path)[1]

            start_dt = get_start_dt(i_video_path)
            if start_dt:
                start_dt_str = start_dt.strftime("%Y-%m-%d_%H-%M-%S")
            else:
                # use original filename when failed to get start_dt
                start_dt_str = os.path.splitext(video_filename)[0]

            output_subdir = os.path.join(output_dir, camera_id)
            os.makedirs(output_subdir, exist_ok=True)
            p = call_proc(i_video_path, output_subdir, camera_id, start_dt_str, encoder)
            processes.append(p)
            if (len(file_list) - 1) > video_count:
                video_count += 1
            else:
                more_video = False
                break
        
        for p in processes:
            p['proc'].wait()
            if p['proc'].returncode != 0:
                print(f'ffmpeg was exited with error. See log files for detail: {p["filestem"]}.log')


        # clear workers
        process = []

    elapsed_time = (datetime.now() - dt_now)
    print(f'ended converting: {datetime.now()},  ellapsed {elapsed_time.total_seconds()/60} minuites')