# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np
import pickle
import argparse

import chardet

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import textwrap

from scipy import stats

import eeg_tools as et
import stage


def make_mouse_info_df(faster_dir_list):
    mouse_info_df = pd.DataFrame()
    epoch_num_stored = None
    for faster_dir in faster_dir_list:
        data_dir = os.path.join(faster_dir, 'data')

        exp_info_df = stage.read_exp_info(data_dir)
        epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime = stage.interpret_exp_info(
            exp_info_df)
        if (epoch_num_stored != None) and epoch_num != epoch_num_stored:
            raise(ValueError('epoch number must be equal among the all dataset'))
        else:
            epoch_num_stored = epoch_num
        
        m_info = stage.read_mouse_info(data_dir)
        m_info['Experiment label'] = exp_label
        m_info['FASTER_DIR'] = faster_dir
        mouse_info_df = pd.concat([mouse_info_df, m_info]) 
    return (mouse_info_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--faster2_dirs", required=True, nargs="*",
                        help="paths to the FASTER2 directoryies")

    args = parser.parse_args()

    faster_dir_list = args.faster2_dirs

    mouse_info_df = make_mouse_info_df(faster_dir_list)

    for i, r in mouse_info_df.iterrows():
        device_id = r[0]

        plot_timeseries_a_mouse(
            data_dir, stage_dir, stage_dir, device_id, sample_freq, epoch_num, start_datetime)
