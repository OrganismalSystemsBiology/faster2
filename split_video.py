# -*- coding: utf-8 -*-
# this scripty requires standalone ffmepg

import argparse
import subprocess
from datetime import datetime, timedelta
import re
import os
import sys
import io

ENCODER = "h264_nvenc"

def interpret_datetimestr(datetime_str):
    """ Find a datetime string and convert it to a datatime object 
    allowing some variant forms
    
    Args:
        datetime_str (string): a string containing datetime
    
    Returns:
        a datetime object
    
    Raises:
        ValueError: raised when interpretation is failed
    """

    datestr_patterns = [r'(\d{4})(\d{2})(\d{2})',
                        r'(\d{4})/(\d{1,2})/(\d{1,2})',
                        r'(\d{4})-(\d{1,2})-(\d{1,2})']

    timestr_patterns = [r'(\d{2})(\d{2})(\d{2})',
                        r'(\d{1,2}):(\d{1,2}):(\d{1,2})',
                        r'(\d{1,2})-(\d{1,2})-(\d{1,2})']

    datetime_obj = None
    for pat in datestr_patterns:
        matched = re.search(pat, datetime_str)
        if matched:
            year = int(matched.group(1))
            month = int(matched.group(2))
            day = int(matched.group(3))
            datetime_str = re.sub(pat, '', datetime_str)

            for pat_time in timestr_patterns:
                matched_time = re.search(pat_time, datetime_str)
                if matched_time:
                    hour = int(matched_time.group(1))
                    minuite = int(matched_time.group(2))
                    second = int(matched_time.group(3))
                    datetime_obj = datetime(year, month, day,
                                            hour, minuite, second)
                    break
            if not matched_time:
                datetime_obj = datetime(year, month, day)
    if not datetime_obj:
        raise ValueError(
            'failed to interpret datetime string \'{}\''.format(datetime_str))

    return(datetime_obj)

def call_proc(clip_count, output_dir):
    start_time = clip_length * clip_count
    clip_start_dt = start_datetime + timedelta(seconds=start_time)
    output_filestem = f'{clip_name}_{clip_start_dt.strftime("%Y-%m-%d_%H-%M-%S")}'
    output_filepath = os.path.join(
        output_dir, output_filestem + '.mp4')
    cmd = ['ffmpeg', '-i', input_video, '-c:v', ENCODER, '-an',
            '-ss', str(start_time), '-t', str(clip_length), '-b:v', '0', '-cq', '32', output_filepath, '-y']
    print(' '.join(cmd))
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    with open(os.path.join(output_dir, 'logs', output_filestem + '.log'), 'w') as logout:
        proc = subprocess.Popen(cmd, stdout=logout, stderr=logout, text=True)
    return({'proc':proc, 'filestem':output_filestem})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_video", required=True, help="path to the input video file")
    parser.add_argument("-o", "--output_dir", required=True, help="path to the directory of the resulting video clips")
    parser.add_argument("-s", "--start_datetime", help="a string of start datetime of the input video (e.g. 2019-07-13_05-33-52)")
    parser.add_argument("-l", "--clip_length", help="duration [sec] of the resulting video clip", type=int, default=300)
    parser.add_argument("-n", "--clip_name", help="a common label of the resulting clips", default="clip")
    parser.add_argument("-w", "--workers", help="the number of parallel processes", type=int, default=1)

    args = parser.parse_args()

    input_video = args.input_video
    output_dir = args.output_dir
    clip_length = args.clip_length
    clip_name = args.clip_name
    start_datetime_opt = args.start_datetime
    worker_num = args.workers

    # try to get the start datetime of input video
    if start_datetime_opt:
        try:
            start_datetime = interpret_datetimestr(start_datetime_opt)
        except ValueError:
            print(f'Failed to interpret the start datetime given by -s option "{start_datetime_opt}"')
    else:
        try:
            filename = os.path.basename(input_video)
            start_datetime = interpret_datetimestr(filename)
        except ValueError as e:
            print(f'Failed to interpret the start datetime from the filename "{filename}"')
            print('exiting')
            sys.exit(1)
    
    clip_count = 0
    more_clip = True
    processes =[]
    while more_clip:
        for i in range(worker_num):
            p = call_proc(clip_count, output_dir)
            processes.append(p)
            clip_count += 1

        for p in processes:
            p['proc'].wait()

            if p['proc'].returncode != 0:
                print('ffmpeg was exited with error. See log files for detail.')
                sys.exit(1)

            logfile_path = os.path.join(output_dir, 'logs', p['filestem'] + '.log')
            with open(logfile_path, 'r') as f:
                last_line_of_log = f.readlines()[-1]
                m = re.search(r'Output file is empty', last_line_of_log)
                if m:
                    excess_filename = p['filestem'] + '.avi'
                    print('The end of the input video was detected. Deleting: ' + excess_filename)
                    more_clip = False
                    os.remove(os.path.join(output_dir, excess_filename))
                
        # clear workers
        process = []