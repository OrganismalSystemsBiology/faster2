# -*- coding: utf-8 -*-
import argparse
import subprocess
from datetime import datetime
import os
from glob import glob
import re
import sys
import stage


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
    with open(os.path.join(output_dir, 'logs', output_filestem + '.log'), 'w') as logout:
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
        start_datetime = stage.interpret_datetimestr(filename)
    except ValueError:
        print(f'[warning] Failed to interpret the start datetime from the filename "{filename}"')
        start_datetime = None

    return start_datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_dir", required=True, help="path to the target directory")
    parser.add_argument("-o", "--output_dir", required=True, help="path to the directory of the resulting video clips")
    parser.add_argument("-w", "--worker", help="the number of parallel workers", type=int, default=1)
    parser.add_argument("-e", "--encoder", help="a string to be passed to ffmpeg -c:v option for the encoder", default="libx264")

    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    output_dir = os.path.abspath(args.output_dir)
    worker_num = args.worker
    encoder = args.encoder

    # recursively get contents of the target dir
    # assuming the structure of [camera_ids]/[video files]
    file_list = [f for f in glob(os.path.join(target_dir, '**/*')) if re.search(r'.*\.(avi|mp4)', f)]

    os.makedirs(target_dir, exist_ok=True)

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