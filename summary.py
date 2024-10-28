# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np
import pickle
import argparse
import copy
import warnings
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import textwrap

import faster2lib.eeg_tools as et
import faster2lib.summary_psd as sp
import faster2lib.summary_common as sc
import stage

from datetime import datetime
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter

# Update rcParams globally
plt.rcParams.update({'font.family': 'Arial'})


def initialize_logger(log_file):
    logger = getLogger()
    logger.setLevel(logging.INFO)

    file_handler = FileHandler(log_file)
    stream_handler = StreamHandler()

    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)
    handler_formatter = Formatter('%(message)s')
    file_handler.setFormatter(handler_formatter)
    stream_handler.setFormatter(handler_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def print_log(msg):
    if 'log' in globals():
        log.info(msg)
    else:
        print(msg)


def print_log_exception(msg):
    if 'log' in globals():
        log.exception(msg)
    else:
        print(msg)


def collect_mouse_info_df(faster_dir_list, epoch_len_sec, mouse_info_ext=None, stage_ext=None):
    """ collects multiple mouse info

    Arguments:
        faster_dir_list [str] -- a list of paths for FASTER directories
        epoch_len_sec -- The epoch length in the second
        mouse_info_ext -- a sub-extention of the mouse.info.csv
        stage_ext -- a sub-extention of the stage files

    Returns:
        {'mouse_info':pd.DataFrame, 'epoch_num':int, 'sample_freq':np.float} -- A dict of
        dataframe of the concatenated mouse info, the sampling frequency,
        and the number of epochs
    """
    mouse_info_df = pd.DataFrame()
    epoch_num_stored = None
    sample_freq_stored = None
    for faster_dir in faster_dir_list:
        data_dir = os.path.join(faster_dir, 'data')

        exp_info_df = et.read_exp_info(data_dir)
        # not used variable: rack_label, start_datetime, end_datetime
        # pylint: disable=unused-variable
        (epoch_num, sample_freq, exp_label, rack_label, \
            start_datetime, end_datetime) = et.interpret_exp_info(exp_info_df, epoch_len_sec)
        if (epoch_num_stored != None) and epoch_num != epoch_num_stored:
            raise ValueError('epoch number must be equal among the all dataset')
        else:
            epoch_num_stored = epoch_num
        if (sample_freq_stored != None) and sample_freq != sample_freq_stored:
            raise ValueError('sample freq must be equal among the all dataset')
        else:
            sample_freq_stored = sample_freq

        m_info = et.read_mouse_info(data_dir, mouse_info_ext)
        m_info['Experiment label'] = exp_label
        m_info['FASTER_DIR'] = faster_dir
        m_info['exp_start_datetime'] = start_datetime
        mouse_info_df = pd.concat([mouse_info_df, m_info], ignore_index=True)
    return ({'mouse_info': mouse_info_df, 'epoch_num': epoch_num, 'mouse_info_ext':mouse_info_ext,
             'stage_ext':stage_ext, 'sample_freq': sample_freq, 'epoch_len_sec': epoch_len_sec})


def serializable_collected_mouse_info(collected_mouse_info):
    """ Because collected_mouse_info_df includes a dataframe, it needs to 
    be converted to json for serialization.
    """
    cmi = collected_mouse_info.copy()
    cmi['mouse_info'] = cmi['mouse_info'].to_json(orient='table')
    return cmi


def make_summary_stats(mouse_info_df, epoch_range, epoch_len_sec, stage_ext):
    """ make summary statics of each mouse:
            stagetime in a day: how many minuites of stages each mouse spent in a day
            stage time profile: hourly profiles of stages over the recording
            stage circadian profile: hourly profiles of stages over a day
            transition matrix: transition probability matrix among each stage
            sw transitino: Sleep (NREM+REM) and Wake transition probability

    Arguments:
        mouse_info_df {pd.DataFram} -- a dataframe returned by collect_mouse_info_df()
        epoch_range {slice} -- target eopchs to be summarized
        epoch_len_sec {int} -- epoch length in seconds
        stage_ext {str} -- the sub-extention of stage files

    Returns:
        {'stagetime': pd.DataFrame,
        'stagetime_profile': [np.array(2)],
        'stagetime_circadian': [np.array(3)],
        'transmat': [np.array(3)],
        'swtrans': [np.array(2)],
        'swtrans_profile': [[np.array(1), np.array(1)]],
        'epoch_num': int} -- A dict of dataframe and arrays of summary stats
    """
    stagetime_df = pd.DataFrame()
    stagetime_profile_list = []
    stagetime_circadian_profile_list = []
    transmat_list = []
    swtrans_list = []
    swtrans_profile_list = []
    swtrans_circadian_profile_list = []

    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label'].strip()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        stats_report = r['Stats report'].strip().upper()
        note = r['Note']
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']
        if stats_report == 'NO':
            print_log(f'[{i+1}] Skipping stage: {faster_dir} {device_label}')
            continue

        # read a stage file
        print_log(f'[{i+1}] Reading stage: {faster_dir} {device_label} {stage_ext}')
        stage_call = et.read_stages(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)
        stage_call = stage_call[epoch_range]
        epoch_num_in_range = len(stage_call)

        # stagetime in a day
        rem, nrem, wake, unknown = stagetime_in_a_day(stage_call)
        stagetime_df = pd.concat(
            [stagetime_df, pd.DataFrame([[exp_label, mouse_group, mouse_id, device_label,
              rem, nrem, wake, unknown, stats_report, note]])], ignore_index=True)

        # stagetime profile
        stagetime_profile_list.append(stagetime_profile(stage_call, epoch_len_sec))

        # stage circadian profile
        stagetime_circadian_profile_list.append(
            stagetime_circadian_profile(stage_call, epoch_len_sec))

        # transition matrix
        transmat_list.append(transmat_from_stages(stage_call))

        # sw transition
        swtrans_list.append(swtrans_from_stages(stage_call))

        # sw transition profile
        swtrans_profile_list.append(swtrans_profile(stage_call, epoch_len_sec))

        # sw transition profile
        swtrans_circadian_profile_list.append(swtrans_circadian_profile(stage_call, epoch_len_sec))

    if stagetime_df.size == 0:
        # There is nothing for the stats report
        return({})

    stagetime_df.columns = ['Experiment label', 'Mouse group', 'Mouse ID',
                            'Device label', 'REM', 'NREM', 'Wake', 'Unknown',
                            'Stats report', 'Note']

    return({'stagetime': stagetime_df,
            'stagetime_profile': stagetime_profile_list,
            'stagetime_circadian': stagetime_circadian_profile_list,
            'transmat': transmat_list,
            'swtrans': swtrans_list,
            'swtrans_profile': swtrans_profile_list,
            'swtrans_circadian': swtrans_circadian_profile_list,
            'epoch_num_in_range': epoch_num_in_range})


def stagetime_in_a_day(stage_call):
    """Count each stage in the stage_call list and calculate
    the daily stage time in minuites.
    Notice this function assumes that the length of the
    stage_call is multiple of days. Also it assumes that the
    stage calls are CAPITALIZED.

    Arguments:
        stage_call {np.array} -- an array of stage calls (e.g. ['WAKE', 'NREM', ...])

    Returns:
        [tuple] -- A tuple of sleep times (rem, nrem, wake, unknown)
    """
    ndata = len(stage_call)

    rem = 1440*np.sum(stage_call == "REM")/ndata
    nrem = 1440*np.sum(stage_call == "NREM")/ndata
    wake = 1440*np.sum(stage_call == "WAKE")/ndata
    unknown = 1440*np.sum(stage_call == "UNKNOWN")/ndata

    return (rem, nrem, wake, unknown)


def stagetime_profile(stage_call, epoch_len_sec):
    """ hourly profiles of stages over the recording

    Arguments:
        stage_call {np.array} -- an array of stage calls (e.g. ['WAKE',
        'NREM', ...])

    Returns:
        [np.array(3, len(stage_calls))] -- each row corrensponds the
        hourly profiles of stages over the recording (rem, nrem, wake)
    """
    sm = stage_call.reshape(-1, int(3600/epoch_len_sec)
                            )  # 60 min(3600 sec) bin
    rem = np.array([np.sum(s == 'REM')*epoch_len_sec /
                    60 for s in sm])  # unit minuite
    nrem = np.array([np.sum(s == 'NREM')*epoch_len_sec /
                     60 for s in sm])  # unit minuite
    wake = np.array([np.sum(s == 'WAKE')*epoch_len_sec /
                     60 for s in sm])  # unit minuite

    return np.array([rem, nrem, wake])


def stagetime_circadian_profile(stage_call, epoch_len_sec):
    """hourly profiles of stages over a day (circadian profile)

    Arguments:
        stage_call {np.array} -- an array of stage calls (e.g. ['WAKE',
        'NREM', ...])

    Returns:
        [np.array(2,3,24)] -- 1st axis: [mean, sd]
                            x 2nd axis [rem, nrem, wake]
                            x 3rd axis [24 hours]
    """
    # 60 min(3600 sec) bin
    sm = stage_call.reshape(-1, int(3600/epoch_len_sec))
    rem = np.array([np.sum(s == 'REM')*epoch_len_sec /
                    60 for s in sm])  # unit minuite
    nrem = np.array([np.sum(s == 'NREM')*epoch_len_sec /
                     60 for s in sm])  # unit minuite
    wake = np.array([np.sum(s == 'WAKE')*epoch_len_sec /
                     60 for s in sm])  # unit minuite

    rem_mat = rem.reshape(-1, 24)
    nrem_mat = nrem.reshape(-1, 24)
    wake_mat = wake.reshape(-1, 24)

    rem_mean = np.apply_along_axis(np.mean, 0, rem_mat)
    rem_sd = np.apply_along_axis(np.std, 0, rem_mat)
    nrem_mean = np.apply_along_axis(np.mean, 0, nrem_mat)
    nrem_sd = np.apply_along_axis(np.std, 0, nrem_mat)
    wake_mean = np.apply_along_axis(np.mean, 0, wake_mat)
    wake_sd = np.apply_along_axis(np.std, 0, wake_mat)

    return np.array([[rem_mean, nrem_mean, wake_mean], [rem_sd, nrem_sd, wake_sd]])


def swtrans_circadian_profile(stage_call, epoch_len_sec):
    """hourly sleep-wake transitions (Psw and Pws) over a day (circadian profile)

    Arguments:
        stage_call {np.array} -- an array of stage calls (e.g. ['WAKE',
        'NREM', ...])

    Returns:
        [np.array(2,2,24)] -- 1st axis [mean, sd]
                            x 2nd axis [Psw, Pws]
                            x 3rd axis [24 hours]
    """
    # 60 min(3600 sec) bin
    sw = swtrans_profile(stage_call, epoch_len_sec) # 1st axis [Psw, Pws] x 2nd axis [recordec hours e.g. 72 hours]

    psw_mat = sw[0].reshape(-1, 24)
    pws_mat = sw[1].reshape(-1, 24)

    # "RuntimeWarning: Mean of empty slice" may occure here and safely ignorable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psw_mean = np.apply_along_axis(np.nanmean, 0, psw_mat)
        psw_sd = np.apply_along_axis(np.nanstd, 0, psw_mat)
        pws_mean = np.apply_along_axis(np.nanmean, 0, pws_mat)
        pws_sd = np.apply_along_axis(np.nanstd, 0, pws_mat)

    return np.array([[psw_mean, pws_mean], [psw_sd, pws_sd]])


def transmat_from_stages(stages):
    """transition probability matrix among each stage

    Arguments:
        stages {np.array} -- an array of stage calls (e.g. ['WAKE',
        'NREM', ...])

    Returns:
        [np.array(3,3)] -- a 3x3 matrix of transition probabilites.
        Notice the order is REM, WAKE, NREM.
    """
    rr = np.sum((stages[:-1] == 'REM') & (stages[1:] == 'REM'))  # REM -> REM
    rw = np.sum((stages[:-1] == 'REM') & (stages[1:] == 'WAKE'))  # REM -> Wake
    rn = np.sum((stages[:-1] == 'REM') & (stages[1:] == 'NREM'))  # REM -> NREM

    wr = np.sum((stages[:-1] == 'WAKE') & (stages[1:] == 'REM'))  # Wake -> REM
    ww = np.sum((stages[:-1] == 'WAKE') &
                (stages[1:] == 'WAKE'))  # Wake-> Wake
    wn = np.sum((stages[:-1] == 'WAKE') &
                (stages[1:] == 'NREM'))  # Wake-> NREM

    nr = np.sum((stages[:-1] == 'NREM') & (stages[1:] == 'REM'))  # NREM -> REM
    nw = np.sum((stages[:-1] == 'NREM') &
                (stages[1:] == 'WAKE'))  # NREM -> Wake
    nn = np.sum((stages[:-1] == 'NREM') &
                (stages[1:] == 'NREM'))  # NREM -> NREM

    r_trans = rr + rw + rn
    w_trans = wr + ww + wn
    n_trans = nr + nw + nn
    transmat = np.array(
        [
            [rr, rw, rn]/r_trans if r_trans > 0 else [np.nan]*3,
            [wr, ww, wn]/w_trans if w_trans > 0 else [np.nan]*3,
            [nr, nw, nn]/n_trans if n_trans > 0 else [np.nan]*3
        ]
    )

    return transmat


def swtrans_from_stages(stages):
    """Sleep (REM+NREM) <> Wake transition probability matrix
    among each stage

    Arguments:
        stages {np.array} -- an array of stage calls (e.g. ['WAKE',
        'NREM', ...])

    Returns:
        [np.array(2)] -- a 1D array of transition probabilites.
        Notice the order is Psw, Pws.
    """
    rr = np.sum((stages[:-1] == 'REM') & (stages[1:] == 'REM'))  # REM -> REM
    rw = np.sum((stages[:-1] == 'REM') & (stages[1:] == 'WAKE'))  # REM -> Wake
    rn = np.sum((stages[:-1] == 'REM') & (stages[1:] == 'NREM'))  # REM -> NREM

    wr = np.sum((stages[:-1] == 'WAKE') & (stages[1:] == 'REM'))  # Wake -> REM
    ww = np.sum((stages[:-1] == 'WAKE') &
                (stages[1:] == 'WAKE'))  # Wake-> Wake
    wn = np.sum((stages[:-1] == 'WAKE') &
                (stages[1:] == 'NREM'))  # Wake-> NREM

    nr = np.sum((stages[:-1] == 'NREM') & (stages[1:] == 'REM'))  # NREM -> REM
    nw = np.sum((stages[:-1] == 'NREM') &
                (stages[1:] == 'WAKE'))  # NREM -> Wake
    nn = np.sum((stages[:-1] == 'NREM') &
                (stages[1:] == 'NREM'))  # NREM -> NREM

    s_trans = rr + rw + rn + nr + nw + nn
    w_trans = wr + ww + wn
    swtrans = np.array([(rw+nw)/s_trans, (wn+wr)/w_trans])  # Psw, Pws

    return swtrans


def swtrans_from_stage_sss_style(stage_call, epoch_len_sec):
    # filter the stage call
    stage_call_f = filter_short_bout(stage_call)

    # convert to sleep / wake label
    # OUTLIER1,2 are labels used in SSS
    sw_call = np.array(['SLEEP' if (x == 'NREM' or x == 'REM' or x == 'SLEEP')
                        else 'WAKE' if (x != 'UNKNOWN' and x != 'OUTLIER1' and x != 'OUTLIER2')
                        else 'UNKNOWN' for x in stage_call_f])

    # transitions array
    tsw = (sw_call[:-1] == 'SLEEP') & (sw_call[1:] == 'WAKE')  # SLEEP -> WAKE
    tss = (sw_call[:-1] == 'SLEEP') & (sw_call[1:] == 'SLEEP') # SLEEP -> SLEEP
    tws = (sw_call[:-1] == 'WAKE') & (sw_call[1:] == 'SLEEP')  # WAKE -> WAKE
    tww = (sw_call[:-1] == 'WAKE') & (sw_call[1:] == 'WAKE')   # WAKE -> WAKE
    tsw = np.append(tsw, 0)
    tss = np.append(tss, 0)
    tws = np.append(tws, 0)
    tww = np.append(tww, 0)

    # transition binary matrix
    ## 24 hours(86400 sec) bin
    tsw_mat_day = tsw.reshape(-1, int(86400/epoch_len_sec))
    tss_mat_day = tss.reshape(-1, int(86400/epoch_len_sec))
    tws_mat_day = tws.reshape(-1, int(86400/epoch_len_sec))
    tww_mat_day = tww.reshape(-1, int(86400/epoch_len_sec))
    ## 12 hours(43200 sec) bin
    tsw_mat_halfday = tsw.reshape(-1, int(43200/epoch_len_sec))
    tss_mat_halfday = tss.reshape(-1, int(43200/epoch_len_sec))
    tws_mat_halfday = tws.reshape(-1, int(43200/epoch_len_sec))
    tww_mat_halfday = tww.reshape(-1, int(43200/epoch_len_sec))

    # first & second halfday
    n_halfday = tsw_mat_halfday.shape[0]
    tsw_mat_halfday_first = tsw_mat_halfday[np.arange(0, n_halfday, 2), :] # first half day
    tss_mat_halfday_first = tss_mat_halfday[np.arange(0, n_halfday, 2), :]
    tws_mat_halfday_first = tws_mat_halfday[np.arange(0, n_halfday, 2), :]
    tww_mat_halfday_first = tww_mat_halfday[np.arange(0, n_halfday, 2), :]
    tsw_mat_halfday_second = tsw_mat_halfday[np.arange(1, n_halfday, 2), :] # second half day
    tss_mat_halfday_second = tss_mat_halfday[np.arange(1, n_halfday, 2), :]
    tws_mat_halfday_second = tws_mat_halfday[np.arange(1, n_halfday, 2), :]
    tww_mat_halfday_second = tww_mat_halfday[np.arange(1, n_halfday, 2), :]

    # transition count array
    daily_nsw = np.apply_along_axis(np.sum, 1, tsw_mat_day)
    daily_nss = np.apply_along_axis(np.sum, 1, tss_mat_day)
    daily_nws = np.apply_along_axis(np.sum, 1, tws_mat_day)
    daily_nww = np.apply_along_axis(np.sum, 1, tww_mat_day)
    halfdaily_first_nsw = np.apply_along_axis(np.sum, 1, tsw_mat_halfday_first)
    halfdaily_first_nss = np.apply_along_axis(np.sum, 1, tss_mat_halfday_first)
    halfdaily_first_nws = np.apply_along_axis(np.sum, 1, tws_mat_halfday_first)
    halfdaily_first_nww = np.apply_along_axis(np.sum, 1, tww_mat_halfday_first)
    halfdaily_second_nsw = np.apply_along_axis(np.sum, 1, tsw_mat_halfday_second)
    halfdaily_second_nss = np.apply_along_axis(np.sum, 1, tss_mat_halfday_second)
    halfdaily_second_nws = np.apply_along_axis(np.sum, 1, tws_mat_halfday_second)
    halfdaily_second_nww = np.apply_along_axis(np.sum, 1, tww_mat_halfday_second)

    # time matrix
    ## 24 hours(86400 sec) bin
    sleep_epoch_mat_day = (sw_call == 'SLEEP').reshape(-1, int(86400/epoch_len_sec))
    wake_epoch_mat_day = (sw_call == 'WAKE').reshape(-1, int(86400/epoch_len_sec))
    ## 12 hours(43200 sec) bin
    sleep_epoch_mat_halfday = (sw_call == 'SLEEP').reshape(-1, int(43200/epoch_len_sec))
    wake_epoch_mat_halfday = (sw_call == 'WAKE').reshape(-1, int(43200/epoch_len_sec))

    # daily and halfdaily time
    daily_sleep_time = np.apply_along_axis(np.sum, 1, sleep_epoch_mat_day*epoch_len_sec/60)
    daily_wake_time = np.apply_along_axis(np.sum, 1, wake_epoch_mat_day*epoch_len_sec/60)
    halfdaily_sleep_time = np.apply_along_axis(
        np.sum, 1, sleep_epoch_mat_halfday*epoch_len_sec/60)
    halfdaily_wake_time = np.apply_along_axis(
        np.sum, 1, wake_epoch_mat_halfday*epoch_len_sec/60)
    halfdaily_first_sleep_time = halfdaily_sleep_time[np.arange(0, n_halfday, 2)]
    halfdaily_first_wake_time = halfdaily_wake_time[np.arange(0, n_halfday, 2)]
    halfdaily_second_sleep_time = halfdaily_sleep_time[np.arange(1, n_halfday, 2)]
    halfdaily_second_wake_time = halfdaily_wake_time[np.arange(1, n_halfday, 2)]

    # all .day
    pswpws_all_day = _calc_sss_style_trans(daily_nsw, daily_nss,
                                           daily_nws, daily_nww,
                                           daily_sleep_time, daily_wake_time)
    # first halfday
    pswpws_halfday_first = _calc_sss_style_trans(halfdaily_first_nsw, halfdaily_first_nss, 
                                                 halfdaily_first_nws, halfdaily_first_nww, 
                                                 halfdaily_first_sleep_time, halfdaily_first_wake_time)
    #second halfday
    pswpws_halfday_second = _calc_sss_style_trans(halfdaily_second_nsw, halfdaily_second_nss, 
                                                  halfdaily_second_nws, halfdaily_second_nww, 
                                                  halfdaily_second_sleep_time, halfdaily_second_wake_time)

    return {'all_day':pswpws_all_day, 'first_halfday':pswpws_halfday_first, 'second_halfday':pswpws_halfday_second}


def _calc_sss_style_trans(nsw, nss, nws, nww, st, wt):
    denom_sw = np.array(nss + nsw, np.float64)
    denom_ws = np.array(nww + nws, np.float64)
    denom_sw[denom_sw == 0] = np.nan
    denom_ws[denom_ws == 0] = np.nan

    _psw = nsw/denom_sw
    _pws = nws/denom_ws
    psw = np.sum(_psw*st)/np.sum(st)
    pws = np.sum(_pws*wt)/np.sum(wt)

    return (psw, pws)


def swtrans_profile(stage_call, epoch_len_sec):
    """ Profile (two timeseries) of the hourly Psw and Psw

    Args:
        stage_call (np.array(1)): an array of stage calls (e.g. ['WAKE',
        'NREM', ...])

    Returns:
        [np.array(1), np.array(1)]: a list of two np.arrays. Each array contain Psw and Pws.
    """
    stage_call = np.array(['SLEEP' if (x == 'NREM' or x == 'REM')
                           else 'WAKE' if x != 'UNKNOWN' else 'UNKNOWN' for x in stage_call])

    tsw = (stage_call[:-1] == 'SLEEP') & (stage_call[1:] == 'WAKE')  # SLEEP -> WAKE
    tss = (stage_call[:-1] == 'SLEEP') & (stage_call[1:] == 'SLEEP') # SLEEP -> SLEEP
    tws = (stage_call[:-1] == 'WAKE') & (stage_call[1:] == 'SLEEP')  # WAKE -> WAKE
    tww = (stage_call[:-1] == 'WAKE') & (stage_call[1:] == 'WAKE')   # WAKE -> WAKE
    tsw = np.append(tsw, 0)
    tss = np.append(tss, 0)
    tws = np.append(tws, 0)
    tww = np.append(tww, 0)

    tsw_mat = tsw.reshape(-1, int(3600/epoch_len_sec))  # 60 min(3600 sec) bin
    tss_mat = tss.reshape(-1, int(3600/epoch_len_sec))
    tws_mat = tws.reshape(-1, int(3600/epoch_len_sec))
    tww_mat = tww.reshape(-1, int(3600/epoch_len_sec))

    hourly_tsw = np.apply_along_axis(np.sum, 1, tsw_mat) 
    hourly_tss = np.apply_along_axis(np.sum, 1, tss_mat) 
    hourly_tws = np.apply_along_axis(np.sum, 1, tws_mat) 
    hourly_tww = np.apply_along_axis(np.sum, 1, tww_mat) 

    denom = np.array(hourly_tss+hourly_tsw, dtype=np.float64)
    denom[denom==0] = np.nan
    hourly_psw = hourly_tsw/denom

    denom = np.array(hourly_tww+hourly_tws, dtype=np.float64)
    denom[denom==0] = np.nan
    hourly_pws = hourly_tws/denom

    return [hourly_psw, hourly_pws]


def bout_table(stage_call):
    """ 
    Args:
        stage_call (np.array(1)): An array of stage calls such as ['WAKE', 'NREM', 'REM', ...].
        This function works with any set of stage calls.

    Returns:
        pd.DataFrame({'stage':, 'len':, 'start_idx':}): A DataFrame of 3 columms of bouts.
        Each row tells what stage, how long, and the epoch index where the bout starts.
    """

    epoch_len = len(stage_call)

    bidx_trans = stage_call[0:(epoch_len-1)] != stage_call[1:epoch_len]
    bidx_trans = np.append(True, bidx_trans) # the first epoch is always True

    idx_trans = np.where(bidx_trans)[0]
    bout_len = idx_trans[1:len(idx_trans)] - idx_trans[0:(len(idx_trans)-1)]
    bout_len = np.append(bout_len, epoch_len - idx_trans[len(idx_trans)-1])
    bout_stage = stage_call[bidx_trans]

    bout_df = pd.DataFrame({'stage':bout_stage, 'len':bout_len, 'start_idx':idx_trans})

    return bout_df


def filter_short_bout(stage_call, min_bout_len=2):
    """ Removes short bouts of stage.
    Args:
        stage_call (np.arra(1)): An array of stages.
        min_bout_len (int, optional): Bouts shorter than this value are to be removed. Defaults to 2.
        Note: The removal overwrites the short bouts by the adjacent previous epoch's stage.

    Returns:
        np.array(1): An array of stages without the short bouts.
    """
    bout_df = bout_table(stage_call)

    bidx_cond = bout_df['len'] < min_bout_len

    stage_call_filtered = stage_call.copy()
    for _, r in bout_df[bidx_cond].iterrows():
        start_idx = r['start_idx']
        bout_len = r['len']
        if start_idx == 0:
            cover_call = stage_call_filtered[start_idx + bout_len]
        else:
            cover_call = stage_call_filtered[start_idx - 1]
        stage_call_filtered[start_idx:(start_idx+bout_len)] = str(cover_call) # str() is to avoid UnicodeDecodeError probably related to the broadcasting

    return stage_call_filtered


def _set_common_features_stagetime_profile(ax, x_max):
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.grid(dashes=(2, 2))

    light_bar_base = matplotlib.patches.Rectangle(
        xy=[0, -8], width=x_max, height=6, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = matplotlib.patches.Rectangle(
            xy=[24*day, -8], width=12, height=6, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)

    ax.set_ylim(-10, 70)


def _set_common_features_swtrans_profile(ax, x_max):
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.grid(dashes=(2, 2))

    light_bar_base = matplotlib.patches.Rectangle(
        xy=[0, -0.08], width=x_max, height=0.06, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = matplotlib.patches.Rectangle(
            xy=[24*day, -0.08], width=12, height=0.06, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)

    ax.set_ylim(-0.1, 0.5)


def _set_common_features_stagetime_profile_rem(ax, x_max):
    r = 4  # a scale factor for y-axis
    ax.set_yticks(np.array([0, 20, 40, 60])/r)
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.grid(dashes=(2, 2))

    light_bar_base = matplotlib.patches.Rectangle(
        xy=[0, -8/r], width=x_max, height=6/r, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = matplotlib.patches.Rectangle(
            xy=[24*day, -8/4], width=12, height=6/r, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)

    ax.set_ylim(-10/r, 70/r)


def draw_stagetime_profile_individual(stagetime_stats, epoch_len_sec, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_profile_list = stagetime_stats['stagetime_profile']
    epoch_num = stagetime_stats['epoch_num_in_range']
    x_max = epoch_num*epoch_len_sec/3600
    x = np.arange(x_max)
    for i, profile in enumerate(stagetime_profile_list):
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(311, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(312, xmargin=0, ymargin=0)
        ax3 = fig.add_subplot(313, xmargin=0, ymargin=0)
        _set_common_features_stagetime_profile_rem(ax1, x_max)
        _set_common_features_stagetime_profile(ax2, x_max)
        _set_common_features_stagetime_profile(ax3, x_max)

        ax1.set_ylabel('Hourly REM\n duration (min)')
        ax2.set_ylabel('Hourly NREM\n duration (min)')
        ax3.set_ylabel('Hourly wake\n duration (min)')
        ax3.set_xlabel('Time (hours)')

        ax1.plot(x, profile[0, :], color=stage.COLOR_REM)
        ax2.plot(x, profile[1, :], color=stage.COLOR_NREM)
        ax3.plot(x, profile[2, :], color=stage.COLOR_WAKE)

        fig.suptitle(
            f'Stage-time profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'stage-time_profile_I_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        sc.savefig(output_dir, filename, fig)


def draw_stagetime_profile_grouped(stagetime_stats, epoch_len_sec, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_profile_list = stagetime_stats['stagetime_profile']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # make stats of stagetime profile: mean and sd over each group
    stagetime_profile_mat = np.array(stagetime_profile_list)  # REM, NREM, Wake
    stagetime_profile_stats_list = []
    for bidx in bidx_group_list:
        stagetime_profile_mean = np.apply_along_axis(
            np.mean, 0, stagetime_profile_mat[bidx])
        stagetime_profile_sd = np.apply_along_axis(
            np.std, 0, stagetime_profile_mat[bidx])
        stagetime_profile_stats_list.append(
            np.array([stagetime_profile_mean, stagetime_profile_sd]))
    epoch_num = stagetime_stats['epoch_num_in_range']
    x_max = epoch_num*epoch_len_sec/3600
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
            csv_df = pd.DataFrame()
            mgs_c = mouse_groups_set[0] # control
            mgs_t = mouse_groups_set[g_idx] # treatment
            num = np.sum(bidx_group_list[g_idx])
            fig = Figure(figsize=(13, 6))
            ax1 = fig.add_subplot(311, xmargin=0, ymargin=0)
            ax2 = fig.add_subplot(312, xmargin=0, ymargin=0)
            ax3 = fig.add_subplot(313, xmargin=0, ymargin=0)

            _set_common_features_stagetime_profile_rem(ax1, x_max)
            _set_common_features_stagetime_profile(ax2, x_max)
            _set_common_features_stagetime_profile(ax3, x_max)

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # REM
            y = stagetime_profile_stats_list[0][0, 0, :]
            y_sem = stagetime_profile_stats_list[0][1, 0, :]/np.sqrt(num_c)
            csv_df = pd.DataFrame({'Time':x, f'{mgs_c}_REM_mean':y, f'{mgs_c}_REM_SEM':y_sem})
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)
            ax1.set_ylabel('Hourly REM\n duration (min)')

            # NREM
            y = stagetime_profile_stats_list[0][0, 1, :]
            y_sem = stagetime_profile_stats_list[0][1, 1, :]/np.sqrt(num_c)
            csv_df = pd.concat([csv_df, pd.DataFrame(
                {f'{mgs_c}_NREM_mean': y, f'{mgs_c}_NREM_SEM': y_sem})], axis=1)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)
            ax2.set_ylabel('Hourly NREM\n duration (min)')

            # Wake
            y = stagetime_profile_stats_list[0][0, 2, :]
            y_sem = stagetime_profile_stats_list[0][1, 2, :]/np.sqrt(num_c)
            csv_df = pd.concat([csv_df, pd.DataFrame(
                {f'{mgs_c}_Wake_mean': y, f'{mgs_c}_Wake_SEM': y_sem})], axis=1)
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem/np.sqrt(num),
                             y + y_sem/np.sqrt(num), color='grey', linewidth=0, alpha=0.3)
            ax3.set_ylabel('Hourly wake\n duration (min)')
            ax3.set_xlabel('Time (hours)')

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            # REM
            y = stagetime_profile_stats_list[g_idx][0, 0, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
            csv_df = pd.concat([csv_df, pd.DataFrame(
                {f'{mgs_t}_REM_mean': y, f'{mgs_t}_REM_SEM': y_sem})], axis=1)
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_REM, linewidth=0, alpha=0.3)

            # NREM
            y = stagetime_profile_stats_list[g_idx][0, 1, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
            csv_df = pd.concat([csv_df, pd.DataFrame(
                {f'{mgs_t}_NREM_mean': y, f'{mgs_t}_NREM_SEM': y_sem})], axis=1)
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

            # Wake
            y = stagetime_profile_stats_list[g_idx][0, 2, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
            csv_df = pd.concat([csv_df, pd.DataFrame(
                {f'{mgs_c}_Wake_mean': y, f'{mgs_c}_Wake_SEM': y_sem})], axis=1)
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)

            fig.suptitle(
                f'{mgs_c} (n={num_c}) v.s. {mgs_t} (n={num})')
            filename = f'stage-time_profile_G_{mgs_c}_vs_{mgs_t}'
            sc.savefig(output_dir, filename, fig)
            csv_df.to_csv(os.path.join(output_dir, f'{filename}.csv'), index=False)
    else:
        # single group
        g_idx = 0

        csv_df = pd.DataFrame()
        mgs_t = mouse_groups_set[g_idx] # treatment
        num = np.sum(bidx_group_list[g_idx])
        x_max = epoch_num*epoch_len_sec/3600
        x = np.arange(x_max)
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(311, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(312, xmargin=0, ymargin=0)
        ax3 = fig.add_subplot(313, xmargin=0, ymargin=0)

        _set_common_features_stagetime_profile_rem(ax1, x_max)
        _set_common_features_stagetime_profile(ax2, x_max)
        _set_common_features_stagetime_profile(ax3, x_max)

        # REM
        y = stagetime_profile_stats_list[g_idx][0, 0, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
        csv_df = pd.DataFrame({'Time':x, f'{mgs_t}_REM_mean':y, f'{mgs_t}_REM_SEM':y_sem})
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_REM, linewidth=0, alpha=0.3)
        ax1.set_ylabel('Hourly REM\n duration (min)')

        # NREM
        y = stagetime_profile_stats_list[g_idx][0, 1, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        csv_df = pd.concat([csv_df, pd.DataFrame(
            {f'{mgs_t}_NREM_mean': y, f'{mgs_t}_NREM_SEM': y_sem})], axis=1)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)
        ax2.set_ylabel('Hourly NREM\n duration (min)')

        # Wake
        y = stagetime_profile_stats_list[g_idx][0, 2, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
        csv_df = pd.concat([csv_df, pd.DataFrame(
            {f'{mgs_t}_Wake_mean': y, f'{mgs_t}_Wake_SEM': y_sem})], axis=1)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem/np.sqrt(num),
                         y + y_sem/np.sqrt(num), color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)
        ax3.set_ylabel('Hourly wake\n duration (min)')
        ax3.set_xlabel('Time (hours)')

        fig.suptitle(f'{mgs_t} (n={num})')
        filename = f'stage-time_profile_G_{mgs_t}'
        sc.savefig(output_dir, filename, fig)
        csv_df.to_csv(os.path.join(output_dir, f'{filename}.csv'), index=False)


def draw_swtrans_profile_individual(stagetime_stats, epoch_len_sec, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_profile_list = stagetime_stats['swtrans_profile']
    epoch_num = stagetime_stats['epoch_num_in_range']
    x_max = epoch_num*epoch_len_sec/3600
    x = np.arange(x_max)
    for i, profile in enumerate(swtrans_profile_list):
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(211, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(212, xmargin=0, ymargin=0)
        _set_common_features_swtrans_profile(ax1, x_max)
        _set_common_features_swtrans_profile(ax2, x_max)

        ax1.set_ylabel('Hourly Psw')
        ax2.set_ylabel('Hourly Pws')
        ax2.set_xlabel('Time (hours)')

        ax1.plot(x, profile[0], color=stage.COLOR_NREM)
        ax2.plot(x, profile[1], color=stage.COLOR_WAKE)

        fig.suptitle(
            f'Sleep-wake transition (Psw Pws) profile:\n{"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'sleep-wake-transition_profile_I_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        sc.savefig(output_dir, filename, fig)


def draw_swtrans_profile_grouped(stagetime_stats, epoch_len_sec, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_profile_list = stagetime_stats['swtrans_profile']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # make stats of stagetime profile: mean and sd over each group
    swtrans_profile_mat = np.array(swtrans_profile_list)  # Psw, Pws
    swtrans_profile_stats_list = []
    for bidx in bidx_group_list:
        # "RuntimeWarning: Mean of empty slice" may occure here and safely ignorable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            swtrans_profile_mean = np.apply_along_axis(
                np.nanmean, 0, swtrans_profile_mat[bidx])
            swtrans_profile_sd = np.apply_along_axis(
                np.nanstd, 0, swtrans_profile_mat[bidx])
            swtrans_profile_stats_list.append(
                np.array([swtrans_profile_mean, swtrans_profile_sd]))
    epoch_num = stagetime_stats['epoch_num_in_range']
    x_max = epoch_num*epoch_len_sec/3600
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
            num = np.sum(bidx_group_list[g_idx])
            fig = Figure(figsize=(13, 6))
            ax1 = fig.add_subplot(211, xmargin=0, ymargin=0)
            ax2 = fig.add_subplot(212, xmargin=0, ymargin=0)

            _set_common_features_swtrans_profile(ax1, x_max)
            _set_common_features_swtrans_profile(ax2, x_max)
            ax2.set_xlabel('Time (hours)')

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # Psw
            y = swtrans_profile_stats_list[0][0, 0, :]
            y_sem = swtrans_profile_stats_list[0][1, 0, :]/np.sqrt(num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)
            ax1.set_ylabel('Hourly Psw')

            # Pws
            y = swtrans_profile_stats_list[0][0, 1, :]
            y_sem = swtrans_profile_stats_list[0][1, 1, :]/np.sqrt(num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)
            ax2.set_ylabel('Hourly `Pws')

            # Treatments
            num = np.sum(bidx_group_list[g_idx])
            # Psw
            y = swtrans_profile_stats_list[g_idx][0, 0, :]
            y_sem = swtrans_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_NREM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

            # Pws
            y = swtrans_profile_stats_list[g_idx][0, 1, :]
            y_sem = swtrans_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
            ax2.plot(x, y, color=stage.COLOR_WAKE)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)

            fig.suptitle(
                f'Sleep-wake transition (Psw Pws) profile:\n{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'sleep-wake-transition_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0

        num = np.sum(bidx_group_list[g_idx])
        x_max = epoch_num*epoch_len_sec/3600
        x = np.arange(x_max)
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(211, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(212, xmargin=0, ymargin=0)

        _set_common_features_swtrans_profile(ax1, x_max)
        _set_common_features_swtrans_profile(ax2, x_max)
        ax2.set_xlabel('Time (hours)')
 
        # Psw
        y = swtrans_profile_stats_list[g_idx][0, 0, :]
        y_sem = swtrans_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

        # Pws
        y = swtrans_profile_stats_list[g_idx][0, 1, :]
        y_sem = swtrans_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_WAKE)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)
        ax2.set_xlabel('Time (hours)')

        fig.suptitle(f'Sleep-wake transition (Psw Pws) profile:\n{mouse_groups_set[g_idx]} (n={num})')
        filename = f'sleep-wake-transition_profile_G_{mouse_groups_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)


def draw_stagetime_circadian_profile_indiviudal(stagetime_stats, epoch_len_sec, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_circadian_list = stagetime_stats['stagetime_circadian']
    epoch_num = stagetime_stats['epoch_num_in_range']
    for i, circadian in enumerate(stagetime_circadian_list):
        x_max = 24
        x = np.arange(x_max)
        fig = Figure(figsize=(13, 4))
        fig.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(131, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(132, xmargin=0, ymargin=0)
        ax3 = fig.add_subplot(133, xmargin=0, ymargin=0)

        _set_common_features_stagetime_profile_rem(ax1, x_max)
        _set_common_features_stagetime_profile(ax2, x_max)
        _set_common_features_stagetime_profile(ax3, x_max)
        ax1.set_xlabel('Time (hours)')
        ax2.set_xlabel('Time (hours)')
        ax3.set_xlabel('Time (hours)')
        ax1.set_ylabel('Hourly REM\n duration (min)')
        ax2.set_ylabel('Hourly NREM\n duration (min)')
        ax3.set_ylabel('Hourly wake\n duration (min)')

        num = epoch_num*epoch_len_sec/3600/24

        # REM
        y = circadian[0, 0, :]
        y_sem = circadian[1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_REM, linewidth=0, alpha=0.3)

        # NREM
        y = circadian[0, 1, :]
        y_sem = circadian[1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

        # Wake
        y = circadian[0, 2, :]
        y_sem = circadian[1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)

        fig.suptitle(
            f'Circadian stage-time profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'stage-time_circadian_profile_I_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        sc.savefig(output_dir, filename, fig)


def draw_stagetime_circadian_profile_grouped(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_circadian_profile_list = stagetime_stats['stagetime_circadian']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # make stats of stagetime circadian profile: mean and sd over each group
    # mouse x [mean of REM, NREM, Wake] x 24 hours
    stagetime_circadian_profile_mat = np.array(
        [ms[0] for ms in stagetime_circadian_profile_list])
    stagetime_circadian_profile_stats_list = []
    for bidx in bidx_group_list:
        stagetime_circadian_profile_mean = np.apply_along_axis(
            np.mean, 0, stagetime_circadian_profile_mat[bidx])
        stagetime_circadian_profile_sd = np.apply_along_axis(
            np.std, 0, stagetime_circadian_profile_mat[bidx])
        stagetime_circadian_profile_stats_list.append(
            np.array([stagetime_circadian_profile_mean, stagetime_circadian_profile_sd]))

    x_max = 24
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        for g_idx in range(1, len(mouse_groups_set)):
            fig = Figure(figsize=(13, 4))
            fig.subplots_adjust(wspace=0.3)
            ax1 = fig.add_subplot(131, xmargin=0, ymargin=0)
            ax2 = fig.add_subplot(132, xmargin=0, ymargin=0)
            ax3 = fig.add_subplot(133, xmargin=0, ymargin=0)

            _set_common_features_stagetime_profile_rem(ax1, x_max)
            _set_common_features_stagetime_profile(ax2, x_max)
            _set_common_features_stagetime_profile(ax3, x_max)
            ax1.set_xlabel('Time (hours)')
            ax2.set_xlabel('Time (hours)')
            ax3.set_xlabel('Time (hours)')
            ax1.set_ylabel('Hourly REM\n duration (min)')
            ax2.set_ylabel('Hourly NREM\n duration (min)')
            ax3.set_ylabel('Hourly wake\n duration (min)')

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # REM
            y = stagetime_circadian_profile_stats_list[0][0, 0, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 0, :]/np.sqrt(
                num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)

            # NREM
            y = stagetime_circadian_profile_stats_list[0][0, 1, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 1, :]/np.sqrt(
                num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)

            # Wake
            y = stagetime_circadian_profile_stats_list[0][0, 2, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 2, :]/np.sqrt(
                num_c)
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            # REM
            y = stagetime_circadian_profile_stats_list[g_idx][0, 0, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(
                num)
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_REM, linewidth=0, alpha=0.3)

            # NREM
            y = stagetime_circadian_profile_stats_list[g_idx][0, 1, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(
                num)
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

            # Wake
            y = stagetime_circadian_profile_stats_list[g_idx][0, 2, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 2, :]/np.sqrt(
                num)
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)

            fig.suptitle(
                f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'stage-time_circadian_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0

        num = np.sum(bidx_group_list[g_idx])
        fig = Figure(figsize=(13, 4))
        fig.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(131, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(132, xmargin=0, ymargin=0)
        ax3 = fig.add_subplot(133, xmargin=0, ymargin=0)


        _set_common_features_stagetime_profile_rem(ax1, x_max)
        _set_common_features_stagetime_profile(ax2, x_max)
        _set_common_features_stagetime_profile(ax3, x_max)

        # REM
        y = stagetime_circadian_profile_stats_list[g_idx][0, 0, :]
        y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_REM, linewidth=0, alpha=0.3)
        ax1.set_ylabel('Hourly REM\n duration (min)')

        # NREM
        y = stagetime_circadian_profile_stats_list[g_idx][0, 1, :]
        y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)
        ax2.set_ylabel('Hourly NREM\n duration (min)')

        # Wake
        y = stagetime_circadian_profile_stats_list[g_idx][0, 2, :]
        y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem/np.sqrt(num),
                            y + y_sem/np.sqrt(num), color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)
        ax3.set_ylabel('Hourly wake\n duration (min)')
        ax3.set_xlabel('Time (hours)')

        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'stage-time_circadian_profile_G_{mouse_groups_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)


def draw_swtrans_circadian_profile_individual(stagetime_stats, epoch_len_sec, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_circadian_list = stagetime_stats['swtrans_circadian']
    epoch_num = stagetime_stats['epoch_num_in_range']
    for i, circadian in enumerate(swtrans_circadian_list):
        x_max = 24
        x = np.arange(x_max)
        fig = Figure(figsize=(13, 4))
        fig.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(121, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(122, xmargin=0, ymargin=0)

        _set_common_features_swtrans_profile(ax1, x_max)
        _set_common_features_swtrans_profile(ax2, x_max)
        ax1.set_xlabel('Time (hours)')
        ax2.set_xlabel('Time (hours)')
        ax1.set_ylabel('Hourly Psw')
        ax2.set_ylabel('Hourly Pws')

        num = epoch_num*epoch_len_sec/3600/24

        # Psw
        y = circadian[0, 0, :]
        y_sem = circadian[1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

        # Pws
        y = circadian[0, 1, :]
        y_sem = circadian[1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_WAKE)
        ax2.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)
        fig.suptitle(
            f'Circadian sleep-wake-transition profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'sleep-wake-transition_circadian_profile_I_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        sc.savefig(output_dir, filename, fig)


def draw_swtrans_circadian_profile_grouped(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_circadian_profile_list = stagetime_stats['swtrans_circadian']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # make stats of stagetime circadian profile: mean and sd over each group
    # mouse x [mean of Psw, Pws] x 24 hours
    swtrans_circadian_profile_mat = np.array(
        [ms[0] for ms in swtrans_circadian_profile_list])
    swtrans_circadian_profile_stats_list = []
    for bidx in bidx_group_list:
        # "RuntimeWarning: Mean of empty slice" may occur here and safely ignorable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            swtrans_circadian_profile_mean = np.apply_along_axis(
                np.nanmean, 0, swtrans_circadian_profile_mat[bidx])
            swtrans_circadian_profile_sd = np.apply_along_axis(
                np.nanstd, 0, swtrans_circadian_profile_mat[bidx])
            swtrans_circadian_profile_stats_list.append(
                np.array([swtrans_circadian_profile_mean, swtrans_circadian_profile_sd]))

    x_max = 24
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        for g_idx in range(1, len(mouse_groups_set)):
            fig = Figure(figsize=(13, 4))
            fig.subplots_adjust(wspace=0.3)
            ax1 = fig.add_subplot(121, xmargin=0, ymargin=0)
            ax2 = fig.add_subplot(122, xmargin=0, ymargin=0)

            _set_common_features_swtrans_profile(ax1, x_max)
            _set_common_features_swtrans_profile(ax2, x_max)
            ax1.set_xlabel('Time (hours)')
            ax2.set_xlabel('Time (hours)')
            ax1.set_ylabel('Hourly Psw')
            ax2.set_ylabel('Hourly Pws')

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # Psw
            y = swtrans_circadian_profile_stats_list[0][0, 0, :]
            y_sem = swtrans_circadian_profile_stats_list[0][1, 0, :]/np.sqrt(
                num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)

            # Pws
            y = swtrans_circadian_profile_stats_list[0][0, 1, :]
            y_sem = swtrans_circadian_profile_stats_list[0][1, 1, :]/np.sqrt(
                num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', linewidth=0, alpha=0.3)

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            # Psw
            y = swtrans_circadian_profile_stats_list[g_idx][0, 0, :]
            y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(
                num)
            ax1.plot(x, y, color=stage.COLOR_NREM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)

            # Pws
            y = swtrans_circadian_profile_stats_list[g_idx][0, 1, :]
            y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(
                num)
            ax2.plot(x, y, color=stage.COLOR_WAKE)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, linewidth=0, alpha=0.3)

            fig.suptitle(
                f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'sleep-wake-transition_circadian_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0

        num = np.sum(bidx_group_list[g_idx])
        fig = Figure(figsize=(13, 4))
        fig.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(121, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(122, xmargin=0, ymargin=0)

        _set_common_features_swtrans_profile(ax1, x_max)
        _set_common_features_swtrans_profile(ax2, x_max)

        # Psw
        y = swtrans_circadian_profile_stats_list[g_idx][0, 0, :]
        y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)
        ax1.set_ylabel('Hourly Psw')

        # Pws
        y = swtrans_circadian_profile_stats_list[g_idx][0, 1, :]
        y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, linewidth=0, alpha=0.3)
        ax2.set_ylabel('Hourly Pws')


        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'sleep-wake-transition_circadian_profile_G_{mouse_groups_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)


def draw_stagetime_barchart(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(10, 4))
    fig.subplots_adjust(wspace=0.5)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    w = 0.8  # bar width
    x_pos = range(num_groups)
    xtick_str_list = ['\n'.join(textwrap.wrap(mouse_groups_set[g_idx], 8))
                      for g_idx in range(num_groups)]
    ax1.set_xticks(x_pos)
    ax2.set_xticks(x_pos)
    ax3.set_xticks(x_pos)
    ax1.set_xticklabels(xtick_str_list)
    ax2.set_xticklabels(xtick_str_list)
    ax3.set_xticklabels(xtick_str_list)
    ax1.set_ylabel('REM duration (min)')
    ax2.set_ylabel('NREM duration (min)')
    ax3.set_ylabel('Wake duration (min)')

    if num_groups > 1:
        # REM
        values_c = stagetime_df['REM'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax1.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos[0], values_c)
        for g_idx in range(1, num_groups):
            values_t = stagetime_df['REM'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            sc.scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        # NREM
        values_c = stagetime_df['NREM'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax2.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax2, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = stagetime_df['NREM'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            sc.scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

        # Wake
        values_c = stagetime_df['Wake'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax3.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax3, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = stagetime_df['Wake'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax3.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            sc.scatter_datapoints(ax3, w, x_pos[g_idx], values_t)
    else:
        # single group
        g_idx = 0
        # REM
        values_t = stagetime_df['REM'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        # NREM
        values_t = stagetime_df['NREM'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        sc.scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

        # Wake
        values_t = stagetime_df['Wake'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax3.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        sc.scatter_datapoints(ax3, w, x_pos[g_idx], values_t)

    fig.suptitle('Stage-times')
    filename = 'stage-time_barchart'
    sc.savefig(output_dir, filename, fig)


def _draw_transition_barchart(mouse_groups, transmat_mat):
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(12, 8))
    fig.subplots_adjust(wspace=0.2)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    w = 0.8  # bar width
    w_sf = 2 / num_groups # scale factor for the bar width
    ax1.set_xticks([0, 2, 4])
    ax2.set_xticks([0, 2])
    ax3.set_xticks([0, 2])
    ax4.set_xticks([0, 2])
    ax1.set_xticklabels(['RR', 'NN', 'WW'])
    ax2.set_xticklabels(['RN', 'RW'])
    ax3.set_xticklabels(['NR', 'NW'])
    ax4.set_xticklabels(['WR', 'WN'])

    # control group (always index: 0)
    num_c = np.sum(bidx_group_list[0])
    rr_vals_c = transmat_mat[bidx_group_list[0]][:, 0, 0]
    ww_vals_c = transmat_mat[bidx_group_list[0]][:, 1, 1]
    nn_vals_c = transmat_mat[bidx_group_list[0]][:, 2, 2]
    rw_vals_c = transmat_mat[bidx_group_list[0]][:, 0, 1]
    rn_vals_c = transmat_mat[bidx_group_list[0]][:, 0, 2]
    wr_vals_c = transmat_mat[bidx_group_list[0]][:, 1, 0]
    wn_vals_c = transmat_mat[bidx_group_list[0]][:, 1, 2]
    nr_vals_c = transmat_mat[bidx_group_list[0]][:, 2, 0]
    nw_vals_c = transmat_mat[bidx_group_list[0]][:, 2, 1]

    # transition from REM may sometime be nan
    rr_vals_c = rr_vals_c[~np.isnan(rr_vals_c)]
    rw_vals_c = rw_vals_c[~np.isnan(rw_vals_c)]
    rn_vals_c = rn_vals_c[~np.isnan(rn_vals_c)]

    if num_groups > 1:
        # staying
        # control
        x_pos = 0 - w + w*w_sf/2 # w*w_sf/2 is just for aligning the bar center
        ax1.bar(x_pos,
                height=np.mean(rr_vals_c),
                yerr=np.std(rr_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos, rr_vals_c)
        x_pos = 2 - w + w*w_sf/2
        ax1.bar(x_pos,
                height=np.mean(nn_vals_c),
                yerr=np.std(nn_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos, nn_vals_c)
        x_pos = 4 - w + w*w_sf/2 
        ax1.bar(x_pos,
                height=np.mean(ww_vals_c),
                yerr=np.std(ww_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos, ww_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            rr_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 0, 0]
            ww_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 1, 1]
            nn_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 2, 2]

            # transition from REM may sometime be nan
            rr_vals_t = rr_vals_t[~np.isnan(rr_vals_t)]

            x_pos = 0 + w*w_sf/2 - w + g_idx*w*w_sf

            ax1.bar(x_pos,
                    height=np.mean(rr_vals_t),
                    yerr=np.std(rr_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            sc.scatter_datapoints(ax1, w, x_pos, rr_vals_t)
            x_pos = 2 + w*w_sf/2 - w + g_idx*w*w_sf
            ax1.bar(x_pos,
                    height=np.mean(nn_vals_t),
                    yerr=np.std(nn_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            sc.scatter_datapoints(ax1, w, x_pos, nn_vals_t)
            x_pos = 4 + w*w_sf/2 - w + g_idx*w*w_sf
            ax1.bar(x_pos,
                    height=np.mean(ww_vals_t),
                    yerr=np.std(ww_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            sc.scatter_datapoints(ax1, w, x_pos, ww_vals_t)

        # Trnsitions from REM
        # control
        x_pos = 0 - w + w*w_sf/2
        ax2.bar(x_pos,
                height=np.mean(rn_vals_c),
                yerr=np.std(rn_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax2, w, x_pos, rn_vals_c)
        x_pos = 2 - w + w*w_sf/2
        ax2.bar(x_pos,
                height=np.mean(rw_vals_c),
                yerr=np.std(rw_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax2, w, x_pos, rw_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            rw_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 0, 1]
            rn_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 0, 2]

            # transition from REM may sometime be nan
            rw_vals_t = rw_vals_t[~np.isnan(rw_vals_t)]
            rn_vals_t = rn_vals_t[~np.isnan(rn_vals_t)]

            x_pos = 0 + w*w_sf/2 - w + g_idx*w*w_sf
            ax2.bar(x_pos,
                    height=np.mean(rn_vals_t),
                    yerr=np.std(rn_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            sc.scatter_datapoints(ax2, w, x_pos, rn_vals_t)
            x_pos = 2 + w*w_sf/2 - w + g_idx*w*w_sf
            ax2.bar(x_pos,
                    height=np.mean(rw_vals_t),
                    yerr=np.std(rw_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            sc.scatter_datapoints(ax2, w, x_pos, rw_vals_t)

        # Trnsitions from NREM
        # control
        x_pos = 0 - w + w*w_sf/2
        ax3.bar(x_pos,
                height=np.mean(nr_vals_c),
                yerr=np.std(nr_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax3, w, x_pos, nr_vals_c)
        x_pos = 2 - w + w*w_sf/2
        ax3.bar(x_pos,
                height=np.mean(nw_vals_c),
                yerr=np.std(nw_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax3, w, x_pos, nw_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            nr_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 2, 0]
            nw_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 2, 1]
            x_pos = 0 + w*w_sf/2 - w + g_idx*w*w_sf
            ax3.bar(x_pos,
                    height=np.mean(nr_vals_t),
                    yerr=np.std(nr_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            sc.scatter_datapoints(ax3, w, x_pos, nr_vals_t)
            x_pos = 2 + w*w_sf/2 - w + g_idx*w*w_sf
            ax3.bar(x_pos,
                    height=np.mean(nw_vals_t),
                    yerr=np.std(nw_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            sc.scatter_datapoints(ax3, w, x_pos, nw_vals_t)

        # Trnsitions from Wake
        # control
        x_pos = 0 - w + w*w_sf/2
        ax4.bar(x_pos,
                height=np.mean(wr_vals_c),
                yerr=np.std(wr_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax4, w, x_pos, wr_vals_c)
        x_pos = 2 - w + w*w_sf/2
        ax4.bar(x_pos,
                height=np.mean(wn_vals_c),
                yerr=np.std(wn_vals_c)/num_c,
                align='center', width=w*w_sf*0.9, capsize=6, color='grey', alpha=0.6)
        sc.scatter_datapoints(ax4, w, x_pos, wn_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            wr_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 1, 0]
            wn_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 1, 2]
            x_pos = 0 + w*w_sf/2 - w + g_idx*w*w_sf
            ax4.bar(x_pos,
                    height=np.mean(wr_vals_t),
                    yerr=np.std(wr_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            sc.scatter_datapoints(ax4, w, x_pos, wr_vals_t)
            x_pos = 2 + w*w_sf/2 - w + g_idx*w*w_sf
            ax4.bar(x_pos,
                    height=np.mean(wn_vals_t),
                    yerr=np.std(wn_vals_t)/num_t,
                    align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            sc.scatter_datapoints(ax4, w, x_pos, wn_vals_t)
    else:
        # staying
        # single group
        x_pos = 0 - w/2
        ax1.bar(x_pos,
                height=np.mean(rr_vals_c),
                yerr=np.std(rr_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos, rr_vals_c)
        x_pos = 2 - w/2
        ax1.bar(x_pos,
                height=np.mean(nn_vals_c),
                yerr=np.std(nn_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos, nn_vals_c)
        x_pos = 4 - w/2
        ax1.bar(x_pos,
                height=np.mean(ww_vals_c),
                yerr=np.std(ww_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        sc.scatter_datapoints(ax1, w, x_pos, ww_vals_c)
        # Trnsitions from REM
        # single group
        x_pos = 0 - w/2
        ax2.bar(x_pos,
                height=np.mean(rn_vals_c),
                yerr=np.std(rn_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        sc.scatter_datapoints(ax2, w, x_pos, rn_vals_c)
        x_pos = 2 - w/2
        ax2.bar(x_pos,
                height=np.mean(rw_vals_c),
                yerr=np.std(rw_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        sc.scatter_datapoints(ax2, w, x_pos, rw_vals_c)
        # Trnsitions from NREM
        # single group
        x_pos = 0 - w/2
        ax3.bar(x_pos,
                height=np.mean(nr_vals_c),
                yerr=np.std(nr_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        sc.scatter_datapoints(ax3, w, x_pos, nr_vals_c)
        x_pos = 2 - w/2
        ax3.bar(x_pos,
                height=np.mean(nw_vals_c),
                yerr=np.std(nw_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        sc.scatter_datapoints(ax3, w, x_pos, nw_vals_c)
        # Trnsitions from Wake
        # single group
        x_pos = 0 - w/2
        ax4.bar(x_pos,
                height=np.mean(wr_vals_c),
                yerr=np.std(wr_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        sc.scatter_datapoints(ax4, w, x_pos, wr_vals_c)
        x_pos = 2 - w/2
        ax4.bar(x_pos,
                height=np.mean(wn_vals_c),
                yerr=np.std(wn_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        sc.scatter_datapoints(ax4, w, x_pos, wn_vals_c)

    return(fig)


def draw_transition_barchart_prob(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    transmat_mat = np.array(stagetime_stats['transmat'])

    fig = _draw_transition_barchart(mouse_groups, transmat_mat)
    axes = fig.axes
    axes[0].set_ylabel('prob. to stay')
    axes[1].set_ylabel('prob. to transit from REM')
    axes[2].set_ylabel('prob. to transit from NREM')
    axes[3].set_ylabel('prob. to transit from Wake')
    fig.suptitle('transition probability')
    filename = f'stage-transition_probability_barchart_{"_".join(mouse_groups_set)}'
    sc.savefig(output_dir, filename, fig)


def _odd(p, epoch_num):
    min_p = 1/epoch_num  # zero probability is replaced by this value
    max_p = 1-1/epoch_num
    pp = min(max(p, min_p), max_p)
    return np.log10(pp/(1-pp))


def draw_transition_barchart_logodds(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    epoch_num = stagetime_stats['epoch_num_in_range']
    transmat_mat = np.array(stagetime_stats['transmat'])
    transmat_mat = np.vectorize(_odd)(transmat_mat, epoch_num)

    fig = _draw_transition_barchart(mouse_groups, transmat_mat)
    axes = fig.axes
    axes[0].set_ylabel('log odds to stay')
    axes[1].set_ylabel('log odds to transit from REM')
    axes[2].set_ylabel('log odds to transit from NREM')
    axes[3].set_ylabel('log odds to transit from Wake')
    fig.suptitle('transition probability (log odds)')
    filename = f'stage-transition_probability_barchart_logodds_{"_".join(mouse_groups_set)}'
    sc.savefig(output_dir, filename, fig)


def _draw_swtransition_barchart(mouse_groups, swtrans_mat):
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    w = 0.8  # bar width
    w_sf = 2 / num_groups # scale factor for the bar width
    ax.set_xticks([0, 2])
    ax.set_xticklabels(['Psw', 'Pws'])

    if num_groups > 1:
        # control group (always index: 0)
        num_c = np.sum(bidx_group_list[0])
        sw_vals_c = swtrans_mat[bidx_group_list[0]][:, 0]
        ws_vals_c = swtrans_mat[bidx_group_list[0]][:, 1]

        ## Psw and Pws
        x_pos = 0 - w + w*w_sf/2 # w*w_sf/2 is just for aligning the bar center
        ax.bar(x_pos,
            height=np.mean(sw_vals_c),
            yerr=np.std(sw_vals_c)/num_c,
            align='center', width=w*w_sf*0.9, capsize=6, color='gray', alpha=0.6)
        sc.scatter_datapoints(ax, w, x_pos, sw_vals_c)
        x_pos = 2 - w + w*w_sf/2 # w*w_sf/2 is just for aligning the bar center
        ax.bar(x_pos,
            height=np.mean(ws_vals_c),
            yerr=np.std(ws_vals_c)/num_c,
            align='center', width=w*w_sf*0.9, capsize=6, color='gray', alpha=0.6)
        sc.scatter_datapoints(ax, w, x_pos, ws_vals_c)

        # test group index: g_idx.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            sw_vals_t = swtrans_mat[bidx_group_list[g_idx]][:, 0]
            x_pos = 0 + w*w_sf/2 - w + g_idx*w*w_sf
            ax.bar(x_pos,
                height=np.mean(sw_vals_t),
                yerr=np.std(sw_vals_t)/num_t,
                align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            sc.scatter_datapoints(ax, w, x_pos, sw_vals_t)

            ws_vals_t = swtrans_mat[bidx_group_list[g_idx]][:, 1]
            x_pos = 2 + w*w_sf/2 - w + g_idx*w*w_sf
            ax.bar(x_pos,
                height=np.mean(ws_vals_t),
                yerr=np.std(ws_vals_t)/num_t,
                align='center', width=w*w_sf*0.9, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            sc.scatter_datapoints(ax, w, x_pos, ws_vals_t)
    else:
        # single group
        g_idx = 0
        num = np.sum(bidx_group_list[g_idx])
        sw_vals = swtrans_mat[bidx_group_list[g_idx]][:, 0]
        x_pos = 0 + g_idx*w/2
        ax.bar(x_pos,
            height=np.mean(sw_vals),
            yerr=np.std(sw_vals)/num,
            align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        sc.scatter_datapoints(ax, w, x_pos, sw_vals)

        ws_vals = swtrans_mat[bidx_group_list[g_idx]][:, 1]
        x_pos = 2 + g_idx*w/2
        ax.bar(x_pos,
            height=np.mean(ws_vals),
            yerr=np.std(ws_vals)/num,
            align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        sc.scatter_datapoints(ax, w, x_pos, ws_vals)

    return(fig)


def draw_swtransition_barchart_prob(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    swtrans_mat = np.array(stagetime_stats['swtrans'])

    fig = _draw_swtransition_barchart(mouse_groups, swtrans_mat)
    axes = fig.axes
    axes[0].set_ylabel('prob. to transit\n between sleep and wake')
    fig.suptitle('sleep/wake trantision probability')
    filename = f'sleep-wake-transition_probability_barchart_{"_".join(mouse_groups_set)}'
    sc.savefig(output_dir, filename, fig)


def draw_swtransition_barchart_logodds(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    epoch_num = stagetime_stats['epoch_num_in_range']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    swtrans_mat = np.array(stagetime_stats['swtrans'])
    swtrans_mat = np.vectorize(_odd)(swtrans_mat, epoch_num)

    fig = _draw_swtransition_barchart(mouse_groups, swtrans_mat)
    axes = fig.axes
    axes[0].set_ylabel('log odds to transit\n between sleep and wake')
    fig.suptitle('sleep/wake trantision probability (log odds)')
    filename = f'sleep-wake-transition_probability_barchart_logodds_{"_".join(mouse_groups_set)}'
    sc.savefig(output_dir, filename, fig)


def logpsd_inv(y, normalizing_fac, normalizing_mean):
    """ inverses the spectrum normalized PSD to get the original PSD. 
    The spectrum normalization is defined as: snorm(log(psd)),
    where log() here means a "decibel like" transformation of 10*np.log10(),
    and snorm() means a standerdization (i.e. mean=0, SD=0) of each frequency
    component of log(PSD). This function implements log^-1(snorm^-1()).

    Arguments:
        y {np.array(freq_bins)} -- spectrum normalized PSD
        normalizing_fac {float} -- SD of each frequency component used for the normalization
        normalizing_mean {float} -- mean of each frequency compenent used for the normalization

    Returns:
        [np_array(freq_bins)] -- original PSD
    """

    return 10**((y / normalizing_fac + normalizing_mean) / 10)


def conv_PSD_from_snorm_PSD(spec_norm):
    """ calculates the conventional PSD from the spectrum normalized PSD matrix.
    The shape of the input PSD matrix is (epoch_num, freq_bins). 

    Arguments:
        spec_norm {'psd': a matrix of spectrum normalized PSD,
                   'norm_fac: an array of factors used to normalize the PSD
                   'mean': an array of means used to normalize the PSD} -- a dict of 
                   spectrum normalized PSD and the associated factors and means.

    Returns:
        [np.array(epoch_num, freq_bins)] -- a conventional PSD matrix
    """

    psd_norm_mat = spec_norm['psd']
    nf = spec_norm['norm_fac']
    nm = spec_norm['mean']
    psd_mat = np.vectorize(logpsd_inv)(psd_norm_mat, nf, nm)

    return psd_mat


def write_sleep_stats(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # mouse_group, stage_type, num, mean, SD, pvalue, star, method
    sleep_stats_df = pd.DataFrame()

    # mouse_group's index:0 is always control
    mg = mouse_groups_set[0]
    bidx = bidx_group_list[0]
    num = np.sum(bidx)
    rem_values_c = stagetime_df['REM'].values[bidx]
    nrem_values_c = stagetime_df['NREM'].values[bidx]
    wake_values_c = stagetime_df['Wake'].values[bidx]
    row1 = [mg, 'REM',  num, np.mean(rem_values_c),  np.std(
        rem_values_c),  np.nan, None, None]
    row2 = [mg, 'NREM', num, np.mean(nrem_values_c), np.std(
        nrem_values_c), np.nan, None, None]
    row3 = [mg, 'Wake', num, np.mean(wake_values_c), np.std(
        wake_values_c), np.nan, None, None]

    sleep_stats_df = pd.concat([sleep_stats_df, pd.DataFrame([row1, row2, row3])], ignore_index=True)
    for i, bidx in enumerate(bidx_group_list[1:]):
        idx = i+1
        mg = mouse_groups_set[idx]
        bidx = bidx_group_list[idx]
        num = np.sum(bidx)
        rem_values_t = stagetime_df['REM'].values[bidx]
        nrem_values_t = stagetime_df['NREM'].values[bidx]
        wake_values_t = stagetime_df['Wake'].values[bidx]

        tr = sc.test_two_sample(rem_values_c,  rem_values_t)  # test for REM
        tn = sc.test_two_sample(nrem_values_c, nrem_values_t)  # test for NREM
        tw = sc.test_two_sample(wake_values_c, wake_values_t)  # test for Wake
        row1 = [mg, 'REM',  num, np.mean(rem_values_t),  np.std(
            rem_values_t),  tr['p_value'], tr['stars'], tr['method']]
        row2 = [mg, 'NREM', num, np.mean(nrem_values_t), np.std(
            nrem_values_t), tn['p_value'], tn['stars'], tn['method']]
        row3 = [mg, 'Wake', num, np.mean(wake_values_t), np.std(
            wake_values_t), tw['p_value'], tw['stars'], tw['method']]

        sleep_stats_df = pd.concat([sleep_stats_df, pd.DataFrame([row1, row2, row3])], ignore_index=True)

    sleep_stats_df.columns = ['Mouse group', 'Stage type',
                              'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    stagetime_df = stagetime_df.round(
        {'REM': 2, 'NREM': 2, 'Wake': 2, 'Unknown': 2})

    sleep_stats_df.to_csv(os.path.join(
        output_dir, 'stage-time_stats_table.csv'), index=False)
    stagetime_df.to_csv(os.path.join(
        output_dir, 'stage-time_table.csv'), index=False)


def write_stagetrans_stats(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # mouse_group, trans_type, num, mean, SD, pvalue, star, method
    transmat_mat = np.array(stagetime_stats['transmat'])
    stagetrans_stats_df = pd.DataFrame()

    # mouse_group's index:0 is always control
    mg = mouse_groups_set[0]
    bidx = bidx_group_list[0]
    num = np.sum(bidx)
    rr_vals_c = transmat_mat[bidx][:, 0, 0]
    ww_vals_c = transmat_mat[bidx][:, 1, 1]
    nn_vals_c = transmat_mat[bidx][:, 2, 2]
    rw_vals_c = transmat_mat[bidx][:, 0, 1]
    rn_vals_c = transmat_mat[bidx][:, 0, 2]
    wr_vals_c = transmat_mat[bidx][:, 1, 0]
    wn_vals_c = transmat_mat[bidx][:, 1, 2]
    nr_vals_c = transmat_mat[bidx][:, 2, 0]
    nw_vals_c = transmat_mat[bidx][:, 2, 1]
    row1 = [mg, 'RR', num, np.mean(rr_vals_c), np.std(rr_vals_c), np.nan, None, None]
    row2 = [mg, 'NN', num, np.mean(nn_vals_c), np.std(nn_vals_c), np.nan, None, None]
    row3 = [mg, 'WW', num, np.mean(ww_vals_c), np.std(ww_vals_c), np.nan, None, None]
    row4 = [mg, 'RN', num, np.mean(rn_vals_c), np.std(rn_vals_c), np.nan, None, None]
    row5 = [mg, 'RW', num, np.mean(rw_vals_c), np.std(rw_vals_c), np.nan, None, None]
    row6 = [mg, 'NR', num, np.mean(nr_vals_c), np.std(nr_vals_c), np.nan, None, None]
    row7 = [mg, 'NW', num, np.mean(nw_vals_c), np.std(nw_vals_c), np.nan, None, None]
    row8 = [mg, 'WR', num, np.mean(wr_vals_c), np.std(wr_vals_c), np.nan, None, None]
    row9 = [mg, 'WN', num, np.mean(wn_vals_c), np.std(wn_vals_c), np.nan, None, None]


    stagetrans_stats_df = pd.concat([stagetrans_stats_df, 
                                     pd.DataFrame([row1, row2, row3, row4, row5, row6, row7, row8, row9])], ignore_index=True)
    for i, bidx in enumerate(bidx_group_list[1:]):
        idx = i+1
        mg = mouse_groups_set[idx]
        bidx = bidx_group_list[idx]
        num = np.sum(bidx)
        rr_vals_t = transmat_mat[bidx][:, 0, 0]
        ww_vals_t = transmat_mat[bidx][:, 1, 1]
        nn_vals_t = transmat_mat[bidx][:, 2, 2]
        rw_vals_t = transmat_mat[bidx][:, 0, 1]
        rn_vals_t = transmat_mat[bidx][:, 0, 2]
        wr_vals_t = transmat_mat[bidx][:, 1, 0]
        wn_vals_t = transmat_mat[bidx][:, 1, 2]
        nr_vals_t = transmat_mat[bidx][:, 2, 0]
        nw_vals_t = transmat_mat[bidx][:, 2, 1]

        trr = sc.test_two_sample(rr_vals_c, rr_vals_t)   
        tnn = sc.test_two_sample(nn_vals_c, nn_vals_t)  
        tww = sc.test_two_sample(ww_vals_c, ww_vals_t)  
        trw = sc.test_two_sample(rw_vals_c, rw_vals_t)  
        trn = sc.test_two_sample(rn_vals_c, rn_vals_t)  
        twr = sc.test_two_sample(wr_vals_c, wr_vals_t)  
        twn = sc.test_two_sample(wn_vals_c, wn_vals_t)  
        tnr = sc.test_two_sample(nr_vals_c, nr_vals_t)  
        tnw = sc.test_two_sample(nw_vals_c, nw_vals_t)  

        row1 = [mg, 'RR', num, np.mean(rr_vals_t), np.std(rr_vals_t), trr['p_value'], trr['stars'], trr['method']]
        row2 = [mg, 'NN', num, np.mean(nn_vals_t), np.std(nn_vals_t), tnn['p_value'], tnn['stars'], tnn['method']]
        row3 = [mg, 'WW', num, np.mean(ww_vals_t), np.std(ww_vals_t), tww['p_value'], tww['stars'], tww['method']]
        row4 = [mg, 'RN', num, np.mean(rn_vals_t), np.std(rn_vals_t), trn['p_value'], trn['stars'], trn['method']]
        row5 = [mg, 'RW', num, np.mean(rw_vals_t), np.std(rw_vals_t), trw['p_value'], trw['stars'], trw['method']]
        row6 = [mg, 'NR', num, np.mean(nr_vals_t), np.std(nr_vals_t), tnr['p_value'], tnr['stars'], tnr['method']]
        row7 = [mg, 'NW', num, np.mean(nw_vals_t), np.std(nw_vals_t), tnw['p_value'], tnw['stars'], tnw['method']]
        row8 = [mg, 'WR', num, np.mean(wr_vals_t), np.std(wr_vals_t), twr['p_value'], twr['stars'], twr['method']]
        row9 = [mg, 'WN', num, np.mean(wn_vals_t), np.std(wn_vals_t), twn['p_value'], twn['stars'], twn['method']]

        stagetrans_stats_df = pd.concat([stagetrans_stats_df, 
            pd.DataFrame([row1, row2, row3, row4, row5, row6, row7, row8, row9])], ignore_index=True)

    stagetrans_stats_df.columns = ['Mouse group', 'Trans type',
                              'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    stagetrans_df = pd.DataFrame(transmat_mat.reshape(-1, 9), columns=['Prr', 'Prw', 'Prn', 'Pwr', 'Pww', 'Pwn', 'Pnr', 'Pnw', 'Pnn'])
    mouse_info_df = stagetime_stats['stagetime'].iloc[:,0:4]
    stagetrans_df = pd.concat([mouse_info_df, stagetrans_df], axis = 1)

    stagetrans_stats_df.to_csv(os.path.join(
        output_dir, 'stage-transition_probability_stats_table.csv'), index=False)
    stagetrans_df.to_csv(os.path.join(
        output_dir, 'stage-transition_probability_table.csv'), index=False)


def write_swtrans_stats(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # mouse_group, stage_type, num, mean, SD, pvalue, star, method
    swtrans_stats_df = pd.DataFrame()
    
    # mouse_group's index:0 is always control
    mg = mouse_groups_set[0]
    bidx = bidx_group_list[0]
    num = np.sum(bidx)
    swtrans_mat = np.array(stagetime_stats['swtrans'])
    psw_values_c = swtrans_mat[bidx, 0]
    pws_values_c = swtrans_mat[bidx, 1]
 
    row1 = [mg, 'Psw',  num, np.mean(psw_values_c),  np.std(
        psw_values_c),  np.nan, None, None]
    row2 = [mg, 'Pws', num, np.mean(pws_values_c), np.std(
        pws_values_c), np.nan, None, None]

    swtrans_stats_df = pd.concat([swtrans_stats_df, 
        pd.DataFrame([row1, row2])], ignore_index=True)
    for i, bidx in enumerate(bidx_group_list[1:]):
        idx = i+1
        mg = mouse_groups_set[idx]
        bidx = bidx_group_list[idx]
        num = np.sum(bidx)
        psw_values_t = swtrans_mat[bidx, 0]
        pws_values_t = swtrans_mat[bidx, 1]

        t_psw = sc.test_two_sample(psw_values_c,  psw_values_t)  # test for Psw
        t_pws = sc.test_two_sample(pws_values_c,  pws_values_t)  # test for Pws
        row1 = [mg, 'Psw',  num, np.mean(psw_values_t),  np.std(
            psw_values_t),  t_psw['p_value'], t_psw['stars'], t_psw['method']]
        row2 = [mg, 'Pws', num, np.mean(pws_values_t), np.std(
            pws_values_t), t_pws['p_value'], t_pws['stars'], t_pws['method']]

        swtrans_stats_df = pd.concat([swtrans_stats_df, pd.DataFrame([row1, row2])], ignore_index=True)

    swtrans_stats_df.columns = ['Mouse group', 'trans type',
                              'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    swtrans_df = pd.DataFrame(swtrans_mat, columns=['Psw', 'Pws'])
    mouse_info_df = stagetime_stats['stagetime'].iloc[:,0:4]
    swtrans_df = pd.concat([mouse_info_df, swtrans_df], axis = 1)

    swtrans_stats_df.to_csv(os.path.join(
        output_dir, 'sleep-wake-transition_probability_stats_table.csv'), index=False)
    swtrans_df.to_csv(os.path.join(
        output_dir, 'sleep-wake-transition_probability_table.csv'), index=False)


def pickle_psd_info_list(psd_info_list, output_dir, filename):
    """ Save the psd_info_list into a file

    Args:
        psd_info_list (list of dict): An object made by make_target_psd_info()
        output_dir: The path to the folder of summary files
        filename (str): The filename of the pickle file
    
    Note: This function assumes the PSD folder already exists
    """
    # Save the psd_info_lists
    pkl_path = os.path.join(output_dir, filename)
    with open(pkl_path, 'wb') as pkl:
        pickle.dump(psd_info_list, pkl)


def process_psd_profile(psd_info_list, epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='', scaling_type='', transform_type='', unit='V'):
    """ Process PSD info to make PSD profiles, do statistical tests and draw plots.
        Args:
            psd_info_list: list of {simple exp info, target info, psd (epoch_num, 129)} for each mouse
            epoch_len_sec: epoch length in seconds
            epoch_range: range of epochs to be analyzed
            sample_freq: sampling frequency
            output_dir: output directory
            psd_type: 'norm' or 'raw' ... voltage distribution type
            scaling_type: 'auc' or 'tdd' or 'none' ... PSD scaling type
            transform_type: 'log' or 'linear' ... PSD transformation type
            unit: unit of PSD
    """
    pre_proc = f'{psd_type}_{scaling_type}_{transform_type}'

    # Mask for the fist halfday
    epoch_num_halfday = int(12*60*60/epoch_len_sec)
    mask_first_halfday = np.tile(
        np.hstack([np.full(epoch_num_halfday, True),
        np.full(epoch_num_halfday, False)]),
        record_day_num)

    # Mask for the second halfday
    mask_second_halfday = np.tile(
        np.hstack([np.full(epoch_num_halfday, False),
        np.full(epoch_num_halfday, True)]),
        record_day_num)

    # PSD profiles (all day)
    psd_profiles_df = sp.make_psd_profile(psd_info_list, sample_freq, psd_type)
    # PSD profiles (first half-day)
    psd_profiles_first_halfday_df = sp.make_psd_profile(
        psd_info_list, sample_freq, psd_type, mask_first_halfday)
    # PSD profiles (second half-day)
    psd_profiles_second_halfday_df = sp.make_psd_profile(
        psd_info_list, sample_freq, psd_type, mask_second_halfday)
    psd_output_dir = os.path.join(output_dir, f'PSD_{psd_type}')

    if scaling_type == 'AUC':
        summary_func = np.sum
    else:
        summary_func = np.mean

    # write a table of PSD (all day)
    sp.write_psd_stats(psd_profiles_df, psd_output_dir, f'{pre_proc}_allday_', summary_func)
    # write a table of PSD (first half-day)
    sp.write_psd_stats(psd_profiles_first_halfday_df, psd_output_dir, f'{pre_proc}_first-halfday_', summary_func)
    # write a table of PSD (second half-day)
    sp.write_psd_stats(psd_profiles_second_halfday_df, psd_output_dir, f'{pre_proc}_second-halfday_', summary_func)

    print_log(f'Drawing the PSDs (voltage distribution, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})')
    # draw PSDs (all day)
    sp.draw_PSDs_individual(psd_profiles_df, sample_freq,
                         f'PSD [{unit}]', psd_output_dir, 
                         psd_type, scaling_type, transform_type, 'allday_')
    sp.draw_PSDs_group(psd_profiles_df, sample_freq,
                    f'PSD [{unit}]', psd_output_dir, 
                    psd_type, scaling_type, transform_type, 'allday_')

    # draw PSDs (first halfday)
    sp.draw_PSDs_individual(psd_profiles_first_halfday_df, sample_freq,
                         f'PSD [{unit}]', psd_output_dir, 
                         psd_type, scaling_type, transform_type, 'first-halfday_')
    sp.draw_PSDs_group(psd_profiles_first_halfday_df, sample_freq,
                    f'PSD [{unit}]', psd_output_dir, 
                    psd_type, scaling_type, transform_type, 'first-halfday_')

    # draw PSDs (second halfday)
    sp.draw_PSDs_individual(psd_profiles_second_halfday_df, sample_freq,
                         f'PSD [{unit}]', psd_output_dir, 
                         psd_type, scaling_type, transform_type, 'second-halfday_')
    sp.draw_PSDs_group(psd_profiles_second_halfday_df, sample_freq,
                    f'PSD [{unit}]', psd_output_dir, 
                    psd_type, scaling_type, transform_type, 'second-halfday_')



def process_psd_timeseries(psd_info_list, epoch_len_sec, epoch_range, sample_freq, output_dir, 
                           psd_type='', scaling_type='', transform_type='', unit='V'):
    freq_bins = sp.psd_freq_bins(sample_freq)
    bidx_delta_freq = sp.get_bidx_delta_freq(freq_bins)
    bidx_all_freq = np.full(len(freq_bins), True)

    print_log('...making the delta-power timeseries in Wake')
    psd_delta_timeseries_wake_df = sp.make_psd_timeseries_df(psd_info_list, epoch_range,  bidx_delta_freq, 'bidx_wake', 
                                                             psd_type, scaling_type,transform_type)
    print_log('...making the delta-power timeseries in NREM')
    psd_delta_timeseries_nrem_df = sp.make_psd_timeseries_df(psd_info_list, epoch_range,  bidx_delta_freq, 'bidx_nrem', 
                                                             psd_type, scaling_type,transform_type)
    print_log('...making the delta-power timeseries in all stages')
    psd_delta_timeseries_df      = sp.make_psd_timeseries_df(psd_info_list, epoch_range,  bidx_delta_freq, None, 
                                                             psd_type, scaling_type,transform_type)
    print_log('...making the total-power timeseries in Wake')
    psd_total_timeseries_wake_df = sp.make_psd_timeseries_df(psd_info_list, epoch_range,  bidx_all_freq, 'bidx_wake', 
                                                             psd_type, scaling_type,transform_type)

    # draw delta-power timeseries
    print_log('...drawing the power timeseries')
    psd_output_dir = os.path.join(output_dir, f'PSD_{psd_type}')

    if transform_type == 'log':
        unit_label = f'log({unit})'
    else:
        unit_label = unit
    
    if scaling_type == 'AUC':
        scale_label = 'AUC-scaled'
    elif scaling_type == 'TDD':
        scale_label = 'TDD-scaled'
    else:
        scale_label = ''
    
    pre_proc = f'{psd_type}_{scaling_type}_{transform_type}'
    # delta in Wake
    psd_delta_timeseries_wake_df.T.to_csv(os.path.join(psd_output_dir, 
                                                       f'power-timeseries_{pre_proc}_delta_Wake.csv'), 
                                                       header=False)
    sp.draw_psd_domain_power_timeseries_individual(psd_delta_timeseries_wake_df, epoch_len_sec, 
                                                   f'Hourly {scale_label} Wake delta power [{unit_label}]', psd_output_dir,
                                                   psd_type, scaling_type, transform_type, 
                                                   'delta', 'Wake_')
    sp.draw_psd_domain_power_timeseries_grouped(psd_delta_timeseries_wake_df, epoch_len_sec, 
                                                f'Hourly {scale_label} Wake delta power [{unit_label}]', psd_output_dir, 
                                                psd_type, scaling_type, transform_type,
                                                'delta', 'Wake_')
    # delta in NREM 
    psd_delta_timeseries_nrem_df.T.to_csv(os.path.join(psd_output_dir, 
                                                       f'power-timeseries_{pre_proc}_delta_NREM.csv'), 
                                                       header=False)
    sp.draw_psd_domain_power_timeseries_individual(psd_delta_timeseries_nrem_df, epoch_len_sec, 
                                                   f'Hourly {scale_label} NREM delta power [{unit_label}]', psd_output_dir, 
                                                   psd_type, scaling_type, transform_type,
                                                   'delta', 'NREM_')
    sp.draw_psd_domain_power_timeseries_grouped(psd_delta_timeseries_nrem_df, epoch_len_sec, 
                                                f'Hourly {scale_label} NREM delta power [{unit_label}]', psd_output_dir, 
                                                psd_type, scaling_type, transform_type,
                                                'delta', 'NREM_')
    # delta in all epoch
    psd_delta_timeseries_df.T.to_csv(os.path.join(psd_output_dir, 
                                                  f'power-timeseries_{pre_proc}_delta.csv'), 
                                                  header=False)
    sp.draw_psd_domain_power_timeseries_individual(psd_delta_timeseries_df, epoch_len_sec, 
                                                   f'Hourly {scale_label} delta power [{unit_label}]', psd_output_dir, 
                                                   psd_type, scaling_type, transform_type,
                                                   'delta')
    sp.draw_psd_domain_power_timeseries_grouped(psd_delta_timeseries_df, epoch_len_sec, 
                                                f'Hourly {scale_label} delta power [{unit_label}]', psd_output_dir, 
                                                psd_type, scaling_type, transform_type,
                                                'delta')
    # total in Wake
    psd_total_timeseries_wake_df.T.to_csv(os.path.join(psd_output_dir, 
                                                       f'power-timeseries_{pre_proc}_total_Wake.csv'), 
                                                       header=False)
    sp.draw_psd_domain_power_timeseries_individual(psd_total_timeseries_wake_df, epoch_len_sec, 
                                                   f'Hourly {scale_label} Wake total power [{unit_label}]', psd_output_dir, 
                                                   psd_type, scaling_type, transform_type,
                                                   'total', 'Wake_')
    sp.draw_psd_domain_power_timeseries_grouped(psd_total_timeseries_wake_df, epoch_len_sec, 
                                                f'Hourly {scale_label} Wake total power [{unit_label}]', psd_output_dir, 
                                                psd_type, scaling_type, transform_type,
                                                'total', 'Wake_')
    
    # PSD-peak plots (Note: lineplot and barplot functions care of writing stats tables)
    sp.draw_psd_peak_circ_heatmap_indiviudal(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, unit_label, psd_output_dir)
    sp.draw_psd_peak_circ_heatmap_grouped(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, unit_label, psd_output_dir)
    sp.draw_psd_peak_circ_lineplot(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, psd_output_dir)
    sp.draw_psd_peak_barplot(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, psd_output_dir)


def make_psd_output_dirs(output_dir, psd_type):
    output_dir = os.path.join(output_dir, f'PSD_{psd_type}')
    os.makedirs(os.path.join(output_dir, 'PDF'), exist_ok=True)


def make_auc_scaled_psd_info_list(psd_info_list):
    auc_psd_info_list = copy.deepcopy(psd_info_list)
    for auc_psd_info in auc_psd_info_list:
        conv_psd_norm = auc_psd_info['norm']
        conv_psd_raw = auc_psd_info['raw']
        auc_psd_norm_mat = np.zeros(conv_psd_norm.shape)
        auc_psd_raw_mat = np.zeros(conv_psd_raw.shape)
        for i, p in enumerate(conv_psd_norm): # row wise
            auc_psd_norm_mat[i,:] = 100*p / np.sum(p) # percentage
        auc_psd_info['norm'] = auc_psd_norm_mat
        for i, p in enumerate(conv_psd_raw): # row wise
            auc_psd_raw_mat[i,:] = 100*p / np.sum(p) # same result with 'norm'
        auc_psd_info['raw'] = auc_psd_raw_mat

    return auc_psd_info_list


def make_tdd_scaled_psd_info(psd_info_list, sample_freq, epoch_len_sec, epoch_num, basal_days):
    # The time domain is from 8- to 12-hour of each basal day 

    # prepare the time-domain binary index
    time_domain_start = 8 # starting time of the reference time domain
    time_domain_end = 12  # ending time of the reference time domain
    bidx_time_domain = np.repeat(False, epoch_num)
    for i in range(basal_days):
        time_domain_range = slice((time_domain_start + 24*i)*3600//epoch_len_sec, 
                                (time_domain_end + 24*i)*3600//epoch_len_sec, None)
        bidx_time_domain[time_domain_range] = True

    # prepare the delta-power frequency binary index
    bidx_delta_freq = sp.get_bidx_delta_freq(sp.psd_freq_bins(sample_freq))

    # calculate the scale for making the delta-power in targeted NREM epochs
    def _calc_scale(psd_mat, bidx_epoch_target, bidx_delta_freq):
        psd_mat_target = psd_mat[bidx_epoch_target, :]
        delta_psd_norm_target = psd_mat_target[:, bidx_delta_freq]
        scale = 1/np.nanmean(np.nanmean(delta_psd_norm_target, axis=1))

        return scale

    tdd_psd_info_list = copy.deepcopy(psd_info_list)
    for tdd_psd_info in tdd_psd_info_list:
        psd_norm = tdd_psd_info['norm']
        psd_raw = tdd_psd_info['raw']

        bidx_epoch_target = tdd_psd_info['bidx_nrem'] & bidx_time_domain
        scale_norm = _calc_scale(psd_norm, bidx_epoch_target, bidx_delta_freq)
        scale_raw = _calc_scale(psd_raw, bidx_epoch_target, bidx_delta_freq)

        tdd_psd_info['norm'] = psd_norm * scale_norm
        tdd_psd_info['raw'] = psd_raw * scale_raw

    return tdd_psd_info_list


def make_log_psd(psd_info_list):
    logpsd_info_list = copy.deepcopy(psd_info_list)
    for logpsd_info in logpsd_info_list:
        logpsd_info['norm'] = 10*np.log10(logpsd_info['norm'])
        logpsd_info['raw'] = 10*np.log10(logpsd_info['raw'])

    return logpsd_info_list


def main(args):
    epoch_len_sec = int(args.epoch_len_sec)

    faster_dir_list = [os.path.abspath(x) for x in args.faster2_dirs]

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = None
    stage_ext = args.stage_ext
    vol_unit = args.unit_voltage
    mouse_info_ext = args.mouse_info_ext

    # set the output directory
    if output_dir is None:
        # default: output to the first FASTER2 directory
        if len(faster_dir_list) > 1:
            basenames = [os.path.basename(dir_path)
                         for dir_path in faster_dir_list]
            path_ext = '_' + '_'.join(basenames)
        else:
            path_ext = ''
        output_dir = os.path.join(faster_dir_list[0], 'summary' + path_ext)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pdf'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)

    global log
    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(output_dir, 'log', f'summary.{dt_str}.log'))
    print_log(f'[{dt_str} - {stage.FASTER2_NAME} - {sys.modules[__name__].__file__}]'
              f' Started in: {os.path.abspath(output_dir)}')

    # collect mouse_infos of the specified (multiple) FASTER dirs
    mouse_info_collected = collect_mouse_info_df(faster_dir_list, epoch_len_sec, mouse_info_ext, stage_ext)
    mouse_info_df = mouse_info_collected['mouse_info']
    epoch_num = mouse_info_collected['epoch_num']
    sample_freq = mouse_info_collected['sample_freq']

    # set the epoch range to be summarized
    if args.epoch_range:
        # use the range given by the command line option
        e_range = [
            int(x.strip()) if x else None for x in args.epoch_range.split(':')]
        epoch_range = slice(*e_range)
    else:
        # default: use the all epochs
        epoch_range = slice(0, epoch_num, None)

    # set the file sub-extention of the stage files to be summarized
    if stage_ext is None:
        # default: 'faster2' for *.faster2.csv
        stage_ext = 'faster2'

    # number of days in the recorded data
    record_day_num = epoch_num * epoch_len_sec / 60 / 60 / 24
    if record_day_num != int(record_day_num):
        raise ValueError(f'The number of recorded days: {record_day_num} must be an integer.\n'
                         f'Check the epoch_num:{epoch_num} and epoch_len_sec:{epoch_len_sec} are correct.')
    else:
        record_day_num = int(record_day_num)

    # set the number of basal days
    if args.basal_days:
        # use the number given by the command line option
        basal_days = int(args.basal_days)
    else:
        basal_days = record_day_num
    print_log(f'Number of recorded days: {record_day_num}, Number of basal days for the time-domain-delta scaling: {basal_days}')

    if basal_days > record_day_num:
        raise ValueError(f'The number of basal days: {basal_days} must be less than or equal to the number of recorded days: {record_day_num}.\n'
                         f'Check the value of basal days command-line option is correct.')

    # add optional information
    mouse_info_collected['epoch_range'] = f'{epoch_range.start}:{epoch_range.stop}'
    mouse_info_collected['stage_ext'] = stage_ext
    mouse_info_collected['basal_days'] = basal_days
    mouse_info_collected['unit_voltage'] = vol_unit

    # dump the collect_mouse_info_df into a file for external scripts
    with open(os.path.join(output_dir, 'collected_mouse_info_df.json'), 'w', encoding='utf8') as outfile:
        json.dump(serializable_collected_mouse_info(mouse_info_collected), outfile)

    # prepare stagetime statistics
    stagetime_stats = make_summary_stats(mouse_info_df, epoch_range, epoch_len_sec, stage_ext)
    if len(stagetime_stats) == 0:
        # quit when there is no stats report
        return -1

    # write tables of sleeptime stats 
    write_sleep_stats(stagetime_stats, output_dir)

    # write tables of sleep-wake transition stats
    write_swtrans_stats(stagetime_stats, output_dir)

    # write tables of stage transition stats
    write_stagetrans_stats(stagetime_stats, output_dir)

    # draw stagetime profile of individual mice
    draw_stagetime_profile_individual(stagetime_stats, epoch_len_sec, output_dir)

    # draw stagetime profile of grouped mice
    draw_stagetime_profile_grouped(stagetime_stats, epoch_len_sec, output_dir)

    # draw stagetime circadian profile of individual mice
    draw_stagetime_circadian_profile_indiviudal(stagetime_stats, epoch_len_sec, output_dir)

    # draw stagetime circadian profile of groups
    draw_stagetime_circadian_profile_grouped(stagetime_stats, output_dir)

    # draw stagetime barchart
    draw_stagetime_barchart(stagetime_stats, output_dir)

    # draw stagetime profile of individual mice
    draw_swtrans_profile_individual(stagetime_stats, epoch_len_sec, output_dir)

    # draw stagetime profile of grouped mice
    draw_swtrans_profile_grouped(stagetime_stats, epoch_len_sec, output_dir)

    # draw stagetime profile of individual mice
    draw_swtrans_circadian_profile_individual(stagetime_stats, epoch_len_sec, output_dir)

    # draw stagetime profile of individual mice
    draw_swtrans_circadian_profile_grouped(stagetime_stats, output_dir)

    # draw stage transition barchart (probability)
    draw_transition_barchart_prob(stagetime_stats, output_dir)

    # draw stage transition barchart (log odds)
    draw_transition_barchart_logodds(stagetime_stats, output_dir)

    # draw sleep/wake transition probability
    draw_swtransition_barchart_prob(stagetime_stats, output_dir)

    # draw sleep/wake transition probability (log odds)
    draw_swtransition_barchart_logodds(stagetime_stats, output_dir)

    # prepare Powerspectrum density (PSD) profiles for individual mice
    # list of {simple exp info, target info, psd (epoch_num, 129)} for each mouse
    psd_info_list = sp.make_target_psd_info(mouse_info_df, epoch_range, epoch_len_sec, sample_freq, stage_ext)

    # Save the psd_info_lists
    print_log('Saving the PSD information')
    pickle_psd_info_list(psd_info_list, output_dir, 'psd_info_list.pkl')

    # log version of psd_info
    print_log('Making the log version of the PSD information')
    logpsd_info_list = make_log_psd(psd_info_list)

    # AUC scaling psd_info for each epoch 
    print_log('Making the area-under-curve (AUC) scaled version of the PSD information')
    auc_psd_info_list = make_auc_scaled_psd_info_list(psd_info_list)
    print_log('Making the log-transformed AUC-scaled version of the PSD information')
    logauc_psd_info_list = make_log_psd(auc_psd_info_list)

    # Time-domain NREM-delta scaling of psd_info
    print_log('Making the time-domin-delta-power (TDD) scaled version of the PSD information')
    tdd_psd_info_list = make_tdd_scaled_psd_info(psd_info_list, sample_freq, epoch_len_sec, epoch_num, basal_days)
    print_log('Making the log-transformed TDD-scaled version of the log-PSD information')
    logtdd_psd_info_list = make_log_psd(tdd_psd_info_list)

    # make output dirs for PSDs
    make_psd_output_dirs(output_dir, 'norm')
    make_psd_output_dirs(output_dir, 'raw')

    # Make PSD stats and plots with different preprocessing
    #   voltage distribution: norm|raw
    #   PSD scaling: none|AUC|TDD
    #   PSD transformation: linear|log

    # norm|raw, none, linear
    process_psd_profile(psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='none', transform_type='linear', unit='AU')
    process_psd_profile(psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='none', transform_type='linear', unit=f'${vol_unit}^{2}/Hz$')
    # norm|raw, none, log
    process_psd_profile(logpsd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='none', transform_type='log', unit='AU')
    process_psd_profile(logpsd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='none', transform_type='log', unit=f'${vol_unit}^{2}/Hz$')
    # norm|raw, AUC, linear
    process_psd_profile(auc_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='AUC', transform_type='linear', unit='%')
    process_psd_profile(auc_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='AUC', transform_type='linear', unit='%')
    # norm|raw, AUC, log
    process_psd_profile(logauc_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                    psd_type='norm', scaling_type='AUC', transform_type='log', unit='log(%)')
    process_psd_profile(logauc_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='AUC', transform_type='log', unit='log(%)')
    # norm|raw, TDD, linear
    process_psd_profile(tdd_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='TDD', transform_type='linear', unit='AU')
    process_psd_profile(tdd_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='TDD', transform_type='linear', unit=f'scaled ${vol_unit}^{2}/Hz$')
    # norm|raw, TDD, log
    process_psd_profile(logtdd_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='TDD', transform_type='log', unit='AU')
    process_psd_profile(logtdd_psd_info_list,epoch_len_sec, record_day_num, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='TDD', transform_type='log', unit=f'log(scaled ${vol_unit}^{2}/Hz$)')

    # PSD timeseries with different preprocessing
    # norm|raw, none, linear
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (norm, none, linear)')
    process_psd_timeseries(psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='none', transform_type='linear', unit='AU')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (raw, none, linear)')
    process_psd_timeseries(psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='none', transform_type='linear', unit=f'${vol_unit}^{2}/Hz$')
    # norm|raw, none, log
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (norm, none, log)')
    process_psd_timeseries(logpsd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='none', transform_type='log', unit='AU')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (raw, none, log)')
    process_psd_timeseries(logpsd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='none', transform_type='log', unit=f'${vol_unit}^{2}/Hz$')
    # norm|raw, AUC, linear
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (norm, AUC, linear)')
    process_psd_timeseries(auc_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='AUC', transform_type='linear', unit='%')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (raw, AUC, linear)')
    process_psd_timeseries(auc_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='AUC', transform_type='linear', unit='%')
    # norm|raw, AUC, log
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (norm, AUC, log)')
    process_psd_timeseries(logauc_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='AUC', transform_type='log', unit='%')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (raw, AUC, log)')
    process_psd_timeseries(logauc_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='AUC', transform_type='log', unit='%')
    # norm|raw, TDD, linear
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (norm, TDD, linear)')
    process_psd_timeseries(tdd_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='TDD', transform_type='linear', unit='AU')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (raw, TDD, linear)')
    process_psd_timeseries(tdd_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='TDD', transform_type='linear', unit=f'scaled ${vol_unit}^{2}/Hz$')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (norm, TDD, log)')
    # norm|raw, TDD, log
    process_psd_timeseries(logtdd_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='norm', scaling_type='TDD', transform_type='log', unit='AU')
    print_log('With the PSD preprocessed as: (voltage distribution, PSD scaling, PSD transformation) = (raw, TDD, log)')
    process_psd_timeseries(logtdd_psd_info_list,epoch_len_sec, epoch_range, sample_freq, output_dir, 
                        psd_type='raw', scaling_type='TDD', transform_type='log', unit=f'scaled ${vol_unit}^{2}/Hz$')

    dt_now = datetime.now()
    print_log(f'[{dt_now} - {sys.modules[__name__].__file__}] Ended')


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-f", "--faster2_dirs", required=True, nargs="*",
                        help="paths to the FASTER2 directories")
    PARSER.add_argument("-e", "--epoch_range",
                        help="a range of epochs to be summaried (default: '0:epoch_num'")
    PARSER.add_argument("-s", "--stage_ext",
                        help="the sub-extention of the stage file (default: faster2)")
    PARSER.add_argument("-m", "--mouse_info_ext",
                        help="the sub-extention of the mouse.info.csv (default: none)")
    PARSER.add_argument("-o", "--output_dir",
                        help="a path to the output files (default: the first FASTER2 directory)")
    PARSER.add_argument("-l", "--epoch_len_sec", help="epoch length in second", default=8)
    PARSER.add_argument("-u", "--unit_voltage", help="The unit of EEG voltage for the raw PSD (default: uV)", default="uV")
    PARSER.add_argument("-b", "--basal_days", help="The number of basal days from recording start", default=3)


    args = PARSER.parse_args()
    try:
        main(args)
    except Exception as e:
        print_log_exception('Unhandled exception occured')
