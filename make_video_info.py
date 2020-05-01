# -*- coding: utf-8 -*-
import argparse
import stage
import os
from glob import glob
import re
import subprocess
import json
import sys
from datetime import datetime, timedelta
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-t", "--target_dir", required=True, help="path to the target directory")
    parser.add_argument("-s", "--start_datetime", help="a string of the start datetime of the input video (e.g. 2020-04-29_19-39-04)")
    args = parser.parse_args()

    target_dir = args.target_dir
    video_start_datetime_opt = args.start_datetime

    # get directories of the target dir i.e. [camera_ids]/
    # assuming the structure of [camera_ids]/[video files]
    dir_list = glob(os.path.join(target_dir, '*'))
    for dir_path in dir_list:
        if not os.path.isdir(dir_path):
            continue
        print('looking into: ', dir_path)

        # get the parent dir name as a camera id
        camera_id = os.path.split(dir_path)[1]

        # get video files
        file_list = [f for f in glob(os.path.join(dir_path, '*')) if re.search(r'.*\.(avi|mp4)', f)]

        # Get the recording start datetime from...
        start_datetime = None
        ## 1. the command line option
        if video_start_datetime_opt:
            try:
                start_datetime = stage.interpret_datetimestr(video_start_datetime_opt)
            except ValueError:
                print(f'[error] Failed to interpret the start datetime given by -s option "{video_start_datetime_opt}"')
                sys.exit(1)

        ## 2. the filename of the first file (overwriting the 1.)
        try:
            filename = os.path.basename(file_list[0])
            start_datetime = stage.interpret_datetimestr(filename)
        except ValueError as e:
            print(f'[warning] Failed to interpret the start datetime from the filename "{filename}"')
        
        ## break the scanning of the directory if it fails to get the start datetime
        if start_datetime == None:
            print("[error] Neither video filename or option didn't give any interpretable datetime")
            print("skipping the directory: " + dir_path)
            break

        # collect video info (start- and end-time)
        video_info_list = []
        for video_path in file_list:
            print('probing video info: ' + video_path)

            cmd = ['ffprobe', '-i', video_path, '-show_streams', '-print_format', 'json', "-loglevel", 'quiet']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                video_info = json.loads(result.stdout)['streams'][0] # a dict of video info
            else:
                print('[error] failed to get video information')
                continue
            
            duration = float(video_info['duration'])
            frame_num, per_secs = video_info['avg_frame_rate'].split('/')
            fps = float(per_secs) / float(frame_num)
            start_str = start_datetime.strftime('%Y-%m-%d_%H-%M-%S.%f')
            start_datetime += timedelta(seconds=duration)
            end_str = start_datetime.strftime('%Y-%m-%d_%H-%M-%S.%f')
            start_datetime += timedelta(seconds=fps)
            video_info_list.append({'filename': os.path.basename(video_path),
                               'start_datetime': start_str,
                               'end_datetime': end_str,
                               'offset': 0.0})
        video_info_df = pd.DataFrame(video_info_list, columns=['filename', 'start_datetime', 'end_datetime', 'offset'])
        video_info_df.to_csv(os.path.join(dir_path, 'video_info.csv'), index=False)