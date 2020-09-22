# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np
import pickle
import argparse
import copy

import chardet

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import textwrap

from scipy import stats

import faster2lib.eeg_tools as et
import stage

from datetime import datetime
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter


DOMAIN_NAMES = ['Slow', 'Delta w/o slow', 'Delta', 'Theta']



def initialize_logger(log_file):
    logger = getLogger(stage.FASTER2_NAME)
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

    return(logger)


def print_log(msg):
    if 'log' in globals():
        log.debug(msg)
    else:
        print(msg)


def collect_mouse_info_df(faster_dir_list):
    """ collects multiple mouse info

    Arguments:
        faster_dir_list [str] -- a list of paths for FASTER directories

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

        exp_info_df = stage.read_exp_info(data_dir)
        # not used variable: rack_label, start_datetime, end_datetime
        # pylint: disable=unused-variable
        epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime = stage.interpret_exp_info(
            exp_info_df)
        if (epoch_num_stored != None) and epoch_num != epoch_num_stored:
            raise(ValueError('epoch number must be equal among the all dataset'))
        else:
            epoch_num_stored = epoch_num
        if (sample_freq_stored != None) and sample_freq != sample_freq_stored:
            raise(ValueError('sample freq must be equal among the all dataset'))
        else:
            sample_freq_stored = sample_freq

        m_info = stage.read_mouse_info(data_dir)
        m_info['Experiment label'] = exp_label
        m_info['FASTER_DIR'] = faster_dir
        mouse_info_df = pd.concat([mouse_info_df, m_info])
    return ({'mouse_info': mouse_info_df, 'epoch_num': epoch_num, 'sample_freq': sample_freq})


def make_summary_stats(mouse_info_df, epoch_range, stage_ext):
    """ make summary statics of each mouse:
            stagetime in a day: how many minuites of stages each mouse spent in a day
            stage time profile: hourly profiles of stages over the recording
            stage circadian profile: hourly profiles of stages over a day
            transition matrix: transition probability matrix among each stage
            sw transitino: Sleep (NREM+REM) and Wake transition probability 

    Arguments:
        mouse_info_df {pd.DataFram} -- a dataframe returned by collect_mouse_info_df()
        epoch_range {slice} -- target eopchs to be summarized
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
            print_log(f'[{i+1}] skipping stage: {faster_dir} {device_label}')
            continue

        # read a stage file
        print_log(f'[{i+1}] reading stage: {faster_dir} {device_label} {stage_ext}')
        stage_call = et.read_stages(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)
        stage_call = stage_call[epoch_range]
        epoch_num = len(stage_call)

        # stagetime in a day
        rem, nrem, wake, unknown = stagetime_in_a_day(stage_call)
        stagetime_df = stagetime_df.append(
            [[exp_label, mouse_group, mouse_id, device_label, rem, nrem, wake, unknown, stats_report, note]], ignore_index=True)

        # stage time profile
        stagetime_profile_list.append(stagetime_profile(stage_call))

        # stage circadian profile
        stagetime_circadian_profile_list.append(
            stagetime_circadian_profile(stage_call))

        # transition matrix
        transmat_list.append(transmat_from_stages(stage_call))

        # sw transition
        swtrans_list.append(swtrans_from_stages(stage_call))

        # sw transition profile
        swtrans_profile_list.append(swtrans_profile(stage_call))

        # sw transition profile
        swtrans_circadian_profile_list.append(swtrans_circadian_profile(stage_call))

    stagetime_df.columns = ['Experiment label', 'Mouse group', 'Mouse ID',
                            'Device label', 'REM', 'NREM', 'Wake', 'Unknown', 'Stats report', 'Note']

    return({'stagetime': stagetime_df,
            'stagetime_profile': stagetime_profile_list,
            'stagetime_circadian': stagetime_circadian_profile_list,
            'transmat': transmat_list,
            'swtrans': swtrans_list,
            'swtrans_profile': swtrans_profile_list,
            'swtrans_circadian': swtrans_circadian_profile_list,
            'epoch_num': epoch_num})


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


def stagetime_profile(stage_call):
    """ hourly profiles of stages over the recording

    Arguments:
        stage_call {np.array} -- an array of stage calls (e.g. ['WAKE', 
        'NREM', ...])

    Returns:
        [np.array(3, len(stage_calls))] -- each row corrensponds the 
        hourly profiles of stages over the recording (rem, nrem, wake)
    """
    sm = stage_call.reshape(-1, int(3600/stage.EPOCH_LEN_SEC)
                            )  # 60 min(3600 sec) bin
    rem = np.array([np.sum(s == 'REM')*stage.EPOCH_LEN_SEC /
                    60 for s in sm])  # unit minuite
    nrem = np.array([np.sum(s == 'NREM')*stage.EPOCH_LEN_SEC /
                     60 for s in sm])  # unit minuite
    wake = np.array([np.sum(s == 'WAKE')*stage.EPOCH_LEN_SEC /
                     60 for s in sm])  # unit minuite

    return np.array([rem, nrem, wake])


def stagetime_circadian_profile(stage_call):
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
    sm = stage_call.reshape(-1, int(3600/stage.EPOCH_LEN_SEC))
    rem = np.array([np.sum(s == 'REM')*stage.EPOCH_LEN_SEC /
                    60 for s in sm])  # unit minuite
    nrem = np.array([np.sum(s == 'NREM')*stage.EPOCH_LEN_SEC /
                     60 for s in sm])  # unit minuite
    wake = np.array([np.sum(s == 'WAKE')*stage.EPOCH_LEN_SEC /
                     60 for s in sm])  # unit minuite

    rem_mat = rem.reshape(-1, 24)
    nrem_mat = nrem.reshape(-1, 24)
    wake_mat = wake.reshape(-1, 24)

    rem_mean = np.apply_along_axis(np.mean, 0, rem_mat)
    rem_sd = np.apply_along_axis(np.std,  0, rem_mat)
    nrem_mean = np.apply_along_axis(np.mean, 0, nrem_mat)
    nrem_sd = np.apply_along_axis(np.std,  0, nrem_mat)
    wake_mean = np.apply_along_axis(np.mean, 0, wake_mat)
    wake_sd = np.apply_along_axis(np.std,  0, wake_mat)

    return np.array([[rem_mean, nrem_mean, wake_mean], [rem_sd, nrem_sd, wake_sd]])


def swtrans_circadian_profile(stage_call):
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
    sw = swtrans_profile(stage_call) # 1st axis [Psw, Pws] x 2nd axis [recordec hours e.g. 72 hours]

    psw_mat = sw[0].reshape(-1, 24)
    pws_mat = sw[1].reshape(-1, 24)
 
    psw_mean = np.apply_along_axis(np.nanmean, 0, psw_mat)
    psw_sd = np.apply_along_axis(np.nanstd,  0, psw_mat)
    pws_mean = np.apply_along_axis(np.nanmean, 0, pws_mat)
    pws_sd = np.apply_along_axis(np.nanstd,  0, pws_mat)

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
        [[rr, rw, rn]/r_trans, [wr, ww, wn]/w_trans, [nr, nw, nn]/n_trans])

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


def swtrans_profile(stage_call):
    """ Profile (two timeseries) of the hourly Psw and Psw

    Args:
        stage_call (np.array(1)): an array of stage calls (e.g. ['WAKE', 
        'NREM', ...])

    Returns:
        [np.array(1), np.array(1)]: a list of two np.arrays. Each array contain Psw and Pws.
    """
    sw_call = np.array(['SLEEP' if (x == 'NREM' or x == 'REM')
                        else 'WAKE' if x != 'UNKNOWN' else 'UNKNOWN' for x in stage_call])

    tsw = (sw_call[:-1] == 'SLEEP') & (sw_call[1:] == 'WAKE')  # SLEEP -> WAKE
    tss = (sw_call[:-1] == 'SLEEP') & (sw_call[1:] == 'SLEEP') # SLEEP -> SLEEP
    tws = (sw_call[:-1] == 'WAKE') & (sw_call[1:] == 'SLEEP')  # WAKE -> WAKE
    tww = (sw_call[:-1] == 'WAKE') & (sw_call[1:] == 'WAKE')   # WAKE -> WAKE
    tsw = np.append(tsw, 0)
    tss = np.append(tss, 0)
    tws = np.append(tws, 0)
    tww = np.append(tww, 0)

    tsw_mat = tsw.reshape(-1, int(3600/stage.EPOCH_LEN_SEC))  # 60 min(3600 sec) bin
    tss_mat = tss.reshape(-1, int(3600/stage.EPOCH_LEN_SEC))
    tws_mat = tws.reshape(-1, int(3600/stage.EPOCH_LEN_SEC))
    tww_mat = tww.reshape(-1, int(3600/stage.EPOCH_LEN_SEC))

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


def test_two_sample(x, y):
    # test.two.sample: Performs two-sample statistical tests according to our labratory's standard.
    ##
    # Arguments:
    # x: first samples
    # y: second samples
    ##
    # Return:
    # A dict of (p.value=p.value, method=method (string))
    ##

    # remove nan
    xx = np.array(x)
    yy = np.array(y)
    xx = xx[~np.isnan(xx)]
    yy = yy[~np.isnan(yy)]

    # If input data length < 2, any test is not applicable.
    if (len(xx) < 2) or (len(yy) < 2):
        p_value = np.nan
        stars = ''
        method = None
    else:
        # If input data length < 3, Shapiro test is not applicable,
        # so we assume false normality of the distribution.
        if (len(xx) < 3) or (len(yy) < 3):
            # Forced rejection of distribution normality
            normality_xx_p = 0
            normality_yy_p = 0
        else:
            normality_xx_p = stats.shapiro(xx)[1]
            normality_yy_p = stats.shapiro(yy)[1]

        equal_variance_p = var_test(xx, yy)['p_value']

        if not ((normality_xx_p < 0.05) or (normality_yy_p < 0.05) or (equal_variance_p < 0.05)):
            # When any null-hypotheses of the normalities of x and of y,
            # and the equal variance of (x,y) are NOT rejected,
            # use Student's t-test
            method = "Student's t-test"
            p_value = stats.ttest_ind(xx, yy, equal_var=True)[1]
        elif not ((normality_xx_p < 0.05) or (normality_yy_p < 0.05)) and (equal_variance_p < 0.05):
            # When null-hypotheses of the normality of x and of y are NOT rejected,
            # but that of the equal variance of (x,y) is rejected,
            # use Welch's t-tet
            method = "Welch's t-test"
            p_value = stats.ttest_ind(xx, yy, equal_var=False)[1]
        else:
            # If none of above was satisfied, use Wilcoxon's ranksum test.
            method = "Wilcoxon test"
            # same as stats.mannwhitneyu() with alternative='two-sided', use_continuity=False
            # or R's wilcox.test(x, y, exact=F, correct=F)
            p_value = stats.ranksums(xx, yy)[1]

        # stars
        if not np.isnan(p_value) and p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = ''

    res = {'p_value': p_value, 'stars': stars, 'method': method}
    return res


def var_test(x, y):
    """ Performs an F test to compare the variances of two samples.
        This function is same as R's var.test()
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    v1 = np.var(y, ddof=1)
    v2 = np.var(x, ddof=1)
    F = v1/v2
    if F > 1:
        p_value = stats.f.sf(F, df2, df1)*2  # two-sided
    else:
        p_value = (1-stats.f.sf(F, df2, df1))*2  # two-sided

    return {'F': F, 'df1': df1, 'df2': df2, 'p_value': p_value}


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


def _set_common_features_delta_power_timeseries(ax, x_max, y_max):
    y_tick_interval = np.power(10, np.ceil(np.log10(y_max))-1)
    ax.set_yticks(np.arange(0, y_max, y_tick_interval))
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.grid(dashes=(2, 2))

    light_bar_base = matplotlib.patches.Rectangle(
        xy=[0, -0.1*y_tick_interval], width=x_max, height=0.1*y_tick_interval, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = matplotlib.patches.Rectangle(
            xy=[24*day, -0.1*y_tick_interval], width=12, height=0.1*y_tick_interval, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)

    ax.set_ylim(-0.15*y_tick_interval, y_max)


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


def _savefig(output_dir, basefilename, fig):
    # JPG
    filename = f'{basefilename}.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0,
                bbox_inches='tight', dpi=100, quality=85, optimize=True)
    # PDF
    filename = f'{basefilename}.pdf'
    fig.savefig(os.path.join(output_dir, 'pdf', filename), pad_inches=0,
                bbox_inches='tight', dpi=100)


def draw_stagetime_profile_individual(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_profile_list = stagetime_stats['stagetime_profile']
    epoch_num = stagetime_stats['epoch_num']
    x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
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
        filename = f'stage-time_profile_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        _savefig(output_dir, filename, fig)


def draw_stagetime_profile_grouped(stagetime_stats, output_dir):
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
    epoch_num = stagetime_stats['epoch_num']
    x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
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
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)
            ax1.set_ylabel('Hourly REM\n duration (min)')

            # NREM
            y = stagetime_profile_stats_list[0][0, 1, :]
            y_sem = stagetime_profile_stats_list[0][1, 1, :]/np.sqrt(num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)
            ax2.set_ylabel('Hourly NREM\n duration (min)')

            # Wake
            y = stagetime_profile_stats_list[0][0, 2, :]
            y_sem = stagetime_profile_stats_list[0][1, 2, :]/np.sqrt(num_c)
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem/np.sqrt(num),
                             y + y_sem/np.sqrt(num), color='grey', alpha=0.3)
            ax3.set_ylabel('Hourly wake\n duration (min)')
            ax3.set_xlabel('Time (hours)')

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            # REM
            y = stagetime_profile_stats_list[g_idx][0, 0, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_REM, alpha=0.3)

            # NREM
            y = stagetime_profile_stats_list[g_idx][0, 1, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            # Wake
            y = stagetime_profile_stats_list[g_idx][0, 2, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(
                f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'stage-time_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            _savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0

        num = np.sum(bidx_group_list[g_idx])
        x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
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
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_REM, alpha=0.3)
        ax1.set_ylabel('Hourly REM\n duration (min)')

        # NREM
        y = stagetime_profile_stats_list[g_idx][0, 1, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax2.set_ylabel('Hourly NREM\n duration (min)')

        # Wake
        y = stagetime_profile_stats_list[g_idx][0, 2, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem/np.sqrt(num),
                         y + y_sem/np.sqrt(num), color=stage.COLOR_WAKE, alpha=0.3)
        ax3.set_ylabel('Hourly wake\n duration (min)')
        ax3.set_xlabel('Time (hours)')

        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'stage-time_profile_G_{mouse_groups_set[g_idx]}'
        _savefig(output_dir, filename, fig)


def draw_swtrans_profile_individual(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_profile_list = stagetime_stats['swtrans_profile']
    epoch_num = stagetime_stats['epoch_num']
    x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
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
        _savefig(output_dir, filename, fig)


def draw_swtrans_profile_grouped(stagetime_stats, output_dir):
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
        swtrans_profile_mean = np.apply_along_axis(
            np.nanmean, 0, swtrans_profile_mat[bidx])
        swtrans_profile_sd = np.apply_along_axis(
            np.nanstd, 0, swtrans_profile_mat[bidx])
        swtrans_profile_stats_list.append(
            np.array([swtrans_profile_mean, swtrans_profile_sd]))
    epoch_num = stagetime_stats['epoch_num']
    x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
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

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # Psw
            y = swtrans_profile_stats_list[0][0, 0, :]
            y_sem = swtrans_profile_stats_list[0][1, 0, :]/np.sqrt(num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)
            ax1.set_ylabel('Hourly Psw')

            # Pws
            y = swtrans_profile_stats_list[0][0, 1, :]
            y_sem = swtrans_profile_stats_list[0][1, 1, :]/np.sqrt(num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)
            ax2.set_ylabel('Hourly `Pws')

            # Treatments
            num = np.sum(bidx_group_list[g_idx])
            # Psw
            y = swtrans_profile_stats_list[g_idx][0, 0, :]
            y_sem = swtrans_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_NREM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            # Pws
            y = swtrans_profile_stats_list[g_idx][0, 1, :]
            y_sem = swtrans_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
            ax2.plot(x, y, color=stage.COLOR_WAKE)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(
                f'Sleep-wake transition (Psw Pws) profile:\n{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'sleep-wake-transition_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            _savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0

        num = np.sum(bidx_group_list[g_idx])
        x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
        x = np.arange(x_max)
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(211, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(212, xmargin=0, ymargin=0)

        _set_common_features_swtrans_profile(ax1, x_max)
        _set_common_features_swtrans_profile(ax2, x_max)
 
        # Psw
        y = swtrans_profile_stats_list[g_idx][0, 0, :]
        y_sem = swtrans_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

        # Pws
        y = swtrans_profile_stats_list[g_idx][0, 1, :]
        y_sem = swtrans_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_WAKE)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)
        ax2.set_xlabel('Time (hours)')

        fig.suptitle(f'Sleep-wake transition (Psw Pws) profile:\n{mouse_groups_set[g_idx]} (n={num})')
        filename = f'sleep-wake-transition_profile_G_{mouse_groups_set[g_idx]}'
        _savefig(output_dir, filename, fig)


def draw_stagetime_circadian_profile_indiviudal(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_circadian_list = stagetime_stats['stagetime_circadian']
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

        num = epoch_num*stage.EPOCH_LEN_SEC/3600/24

        # REM
        y = circadian[0, 0, :]
        y_sem = circadian[1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_REM, alpha=0.3)

        # NREM
        y = circadian[0, 1, :]
        y_sem = circadian[1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

        # Wake
        y = circadian[0, 2, :]
        y_sem = circadian[1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

        fig.suptitle(
            f'Circadian stage-time profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'stage-time_circadian_profile_I_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        _savefig(output_dir, filename, fig)


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
                             y + y_sem, color='grey', alpha=0.3)

            # NREM
            y = stagetime_circadian_profile_stats_list[0][0, 1, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 1, :]/np.sqrt(
                num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)

            # Wake
            y = stagetime_circadian_profile_stats_list[0][0, 2, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 2, :]/np.sqrt(
                num_c)
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            # REM
            y = stagetime_circadian_profile_stats_list[g_idx][0, 0, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(
                num)
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_REM, alpha=0.3)

            # NREM
            y = stagetime_circadian_profile_stats_list[g_idx][0, 1, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(
                num)
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            # Wake
            y = stagetime_circadian_profile_stats_list[g_idx][0, 2, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 2, :]/np.sqrt(
                num)
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(
                f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'stage-time_circadian_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            _savefig(output_dir, filename, fig)
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
                            y + y_sem, color=stage.COLOR_REM, alpha=0.3)
        ax1.set_ylabel('Hourly REM\n duration (min)')

        # NREM
        y = stagetime_circadian_profile_stats_list[g_idx][0, 1, :]
        y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax2.set_ylabel('Hourly NREM\n duration (min)')

        # Wake
        y = stagetime_circadian_profile_stats_list[g_idx][0, 2, :]
        y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem/np.sqrt(num),
                            y + y_sem/np.sqrt(num), color=stage.COLOR_WAKE, alpha=0.3)
        ax3.set_ylabel('Hourly wake\n duration (min)')
        ax3.set_xlabel('Time (hours)')

        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'stage-time_profile_G_{mouse_groups_set[g_idx]}'
        _savefig(output_dir, filename, fig)


def draw_swtrans_circadian_profile_individual(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_circadian_list = stagetime_stats['swtrans_circadian']
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

        num = epoch_num*stage.EPOCH_LEN_SEC/3600/24

        # Psw
        y = circadian[0, 0, :]
        y_sem = circadian[1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

        # Pws
        y = circadian[0, 1, :]
        y_sem = circadian[1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_WAKE)
        ax2.fill_between(x, y - y_sem,
                         y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)
        fig.suptitle(
            f'Circadian sleep-wake-transition profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'sleep-wake-transition_circadian_profile_I_{"_".join(stagetime_df.iloc[i,0:4].values)}'
        _savefig(output_dir, filename, fig)

def draw_swtrans_circadian_profile_grouped(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    swtrans_circadian_profile_list = stagetime_stats['swtrans_circadian']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    # make stats of stagetime circadian profile: mean and sd over each group
    # mouse x [mean of REM, NREM, Wake] x 24 hours
    swtrans_circadian_profile_mat = np.array(
        [ms[0] for ms in swtrans_circadian_profile_list])
    swtrans_circadian_profile_stats_list = []
    for bidx in bidx_group_list:
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
                             y + y_sem, color='grey', alpha=0.3)

            # Pws
            y = swtrans_circadian_profile_stats_list[0][0, 1, :]
            y_sem = swtrans_circadian_profile_stats_list[0][1, 1, :]/np.sqrt(
                num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color='grey', alpha=0.3)

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            # Psw
            y = swtrans_circadian_profile_stats_list[g_idx][0, 0, :]
            y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(
                num)
            ax1.plot(x, y, color=stage.COLOR_NREM)
            ax1.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            # Pws
            y = swtrans_circadian_profile_stats_list[g_idx][0, 1, :]
            y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(
                num)
            ax2.plot(x, y, color=stage.COLOR_WAKE)
            ax2.fill_between(x, y - y_sem,
                             y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(
                f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'sleep-wake-transition_circadian_profile_G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            _savefig(output_dir, filename, fig)
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
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax1.set_ylabel('Hourly Psw')

        # Pws
        y = swtrans_circadian_profile_stats_list[g_idx][0, 1, :]
        y_sem = swtrans_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax2.set_ylabel('Hourly Pws')


        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'sleep-wake-transition_circadian_profile_G_{mouse_groups_set[g_idx]}'
        _savefig(output_dir, filename, fig)


def draw_psd_delta_timeseries_individual(psd_delta_timeseries_df, y_label, output_dir, opt_label=''):
    mouse_groups = psd_delta_timeseries_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]  

    hourly_ts_list = []
    for _, ts in psd_delta_timeseries_df.iloc[:,4:].iterrows():
        ts_mat = ts.to_numpy(dtype=np.float64).reshape(-1, int(3600/stage.EPOCH_LEN_SEC))
        # The rows with all nan needs to be avoided in np.nanmean
        idx_all_nan = np.where([np.all(np.isnan(x)) for x in ts_mat])
        ts_mat[idx_all_nan, :] = 0
        hourly_ts = np.apply_along_axis(np.nanmean, 1, ts_mat)
        hourly_ts[idx_all_nan] = np.nan # matplotlib can handle np.nan
        hourly_ts_list.append(hourly_ts)

    hourly_ts_mat = np.array(hourly_ts_list)

    # this is just for deciding y_max
    delta_timeseries_stats_list=[]
    for bidx in bidx_group_list:
        hourly_ts_mat_group = hourly_ts_mat[bidx]
        idx_all_nan = np.where([np.all(np.isnan(r)) for r in hourly_ts_mat_group.T])
        hourly_ts_mat_group[:, idx_all_nan] = 0 # this is for np.nanmean and np.nanstd
        delta_timeseries_mean = np.apply_along_axis(
            np.nanmean, 0, hourly_ts_mat_group)
        delta_timeseries_sd = np.apply_along_axis(
            np.nanstd, 0, hourly_ts_mat_group)
        delta_timeseries_mean[idx_all_nan] = np.nan
        delta_timeseries_sd[idx_all_nan] = np.nan
        delta_timeseries_stats_list.append(
            np.array([delta_timeseries_mean, delta_timeseries_sd]))
    y_max = np.nanmax(np.array([ts_stats[0] for ts_stats in delta_timeseries_stats_list])) * 1.1

    x_max = ts_mat.shape[0]
    x = np.arange(x_max)
    for i, profile in enumerate(hourly_ts_list):
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)
        _set_common_features_delta_power_timeseries(ax1, x_max, y_max)

        ax1.set_ylabel(y_label)
        ax1.set_xlabel('Time (hours)')

        ax1.plot(x, profile, color=stage.COLOR_NREM)

        fig.suptitle(
            f'Stage-time profile: {"  ".join(psd_delta_timeseries_df.iloc[i,0:4].values)}')

        filename = f'delta_power_timeseries_{opt_label}{"_".join(psd_delta_timeseries_df.iloc[i,0:4].values)}'
        _savefig(output_dir, filename, fig)


def draw_psd_delta_timeseries_grouped(psd_delta_timeseries_df, y_label, output_dir, opt_label=''):
    mouse_groups = psd_delta_timeseries_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]  

    hourly_ts_list = []
    for _, ts in psd_delta_timeseries_df.iloc[:,4:].iterrows():
        ts_mat = ts.to_numpy().reshape(-1, int(3600/stage.EPOCH_LEN_SEC))
        # The rows with all nan needs to be avoided in np.nanmean
        idx_all_nan = np.where([np.all(np.isnan(x)) for x in ts_mat])
        ts_mat[idx_all_nan, :] = 0
        hourly_ts = np.apply_along_axis(np.nanmean, 1, ts_mat)
        hourly_ts[idx_all_nan] = np.nan # matplotlib can handle np.nan
        hourly_ts_list.append(hourly_ts)
    hourly_ts_mat = np.array(hourly_ts_list)

    delta_timeseries_stats_list=[]
    for bidx in bidx_group_list:
        hourly_ts_mat_group = hourly_ts_mat[bidx]
        idx_all_nan = np.where([np.all(np.isnan(r)) for r in hourly_ts_mat_group.T])
        hourly_ts_mat_group[:, idx_all_nan] = 0 # this is for np.nanmean and np.nanstd
        delta_timeseries_mean = np.apply_along_axis(
            np.nanmean, 0, hourly_ts_mat_group)
        delta_timeseries_sd = np.apply_along_axis(
            np.nanstd, 0, hourly_ts_mat_group)
        delta_timeseries_mean[idx_all_nan] = np.nan
        delta_timeseries_sd[idx_all_nan] = np.nan
        delta_timeseries_stats_list.append(
            np.array([delta_timeseries_mean, delta_timeseries_sd]))

    # pylint: disable=E1136  # pylint/issues/3139
    x_max = hourly_ts_mat.shape[1]
    y_max = np.nanmax(np.array([ts_stats[0] for ts_stats in delta_timeseries_stats_list])) * 1.1
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
            num = np.sum(bidx_group_list[g_idx])
            fig = Figure(figsize=(13, 6))
            ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)

            _set_common_features_delta_power_timeseries(ax1, x_max, y_max)

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            y = delta_timeseries_stats_list[0][0, :]
            y_sem = delta_timeseries_stats_list[0][1, :]/np.sqrt(num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                            y + y_sem, color='grey', alpha=0.3)
            ax1.set_ylabel(y_label)
            ax1.set_xlabel('Time (hours)')

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            y = delta_timeseries_stats_list[g_idx][0, :]
            y_sem = delta_timeseries_stats_list[g_idx][1, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_NREM)
            ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            fig.suptitle(
                f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'delta_power_timeseries_{opt_label}{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            _savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0
        num = np.sum(bidx_group_list[g_idx])
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)

        _set_common_features_delta_power_timeseries(ax1, x_max, y_max)

        y = delta_timeseries_stats_list[g_idx][0, :]
        y_sem = delta_timeseries_stats_list[g_idx][1, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax1.set_ylabel(y_label)
        ax1.set_xlabel('Time (hours)')

        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'{opt_label}delta_power_timeseries_{mouse_groups_set[g_idx]}'
        _savefig(output_dir, filename, fig)


def x_shifts(values, y_min, y_max, width):
    #    print_log(y_min, y_max)
    counts, _ = np.histogram(values, range=(
        np.min([y_min, np.min(values)]), np.max([y_max, np.max(values)])), bins=30)
    sorted_values = sorted(values)
    shifts = []
#    print_log(counts)
    non_zero_counts = counts[counts > 0]
    for c in non_zero_counts:
        if c == 1:
            shifts.append(0)
        else:
            p = np.arange(1, c+1)  # point counts
            s = np.repeat(p, 2)[:p.size] * (-1)**p * width / \
                10  # [-1, 1, -2, 2, ...] * width/10
            shifts.extend(s)

#     print_log(shifts)
#     print_log(sorted_values)
    return [np.array(shifts), sorted_values]


def scatter_datapoints(ax, w, x_pos, values):
    s, v = x_shifts(values, *ax.get_ylim(), w)
    ax.scatter(x_pos + s, v, color='darkgrey')


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
        scatter_datapoints(ax1, w, x_pos[0], values_c)
        for g_idx in range(1, num_groups):
            values_t = stagetime_df['REM'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        # NREM
        values_c = stagetime_df['NREM'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax2.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax2, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = stagetime_df['NREM'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

        # Wake
        values_c = stagetime_df['Wake'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax3.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax3, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = stagetime_df['Wake'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax3.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            scatter_datapoints(ax3, w, x_pos[g_idx], values_t)
    else:
        # single group
        g_idx = 0
        # REM
        values_t = stagetime_df['REM'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        # NREM
        values_t = stagetime_df['NREM'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

        # Wake
        values_t = stagetime_df['Wake'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax3.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax3, w, x_pos[g_idx], values_t)

    fig.suptitle('Stage-times')
    filename = 'stage-time_barchart'
    _savefig(output_dir, filename, fig)


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
    nn_vals_c = transmat_mat[bidx_group_list[0]][:, 1, 1]
    ww_vals_c = transmat_mat[bidx_group_list[0]][:, 2, 2]
    rw_vals_c = transmat_mat[bidx_group_list[0]][:, 0, 1]
    rn_vals_c = transmat_mat[bidx_group_list[0]][:, 0, 2]
    wr_vals_c = transmat_mat[bidx_group_list[0]][:, 1, 0]
    wn_vals_c = transmat_mat[bidx_group_list[0]][:, 1, 2]
    nr_vals_c = transmat_mat[bidx_group_list[0]][:, 2, 0]
    nw_vals_c = transmat_mat[bidx_group_list[0]][:, 2, 1]

    if num_groups > 1:
        # staying
        # control
        x_pos = 0 - w/2
        ax1.bar(x_pos,
                height=np.mean(rr_vals_c),
                yerr=np.std(rr_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax1, w, x_pos, rr_vals_c)
        x_pos = 2 - w/2
        ax1.bar(x_pos,
                height=np.mean(nn_vals_c),
                yerr=np.std(nn_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax1, w, x_pos, nn_vals_c)
        x_pos = 4 - w/2
        ax1.bar(x_pos,
                height=np.mean(ww_vals_c),
                yerr=np.std(ww_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax1, w, x_pos, ww_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            rr_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 0, 0]
            nn_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 1, 1]
            ww_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 2, 2]
            x_pos = 0 + g_idx*w/2
            ax1.bar(x_pos,
                    height=np.mean(rr_vals_t),
                    yerr=np.std(rr_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            scatter_datapoints(ax1, w, x_pos, rr_vals_t)
            x_pos = 2 + g_idx*w/2
            ax1.bar(x_pos,
                    height=np.mean(nn_vals_t),
                    yerr=np.std(nn_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            scatter_datapoints(ax1, w, x_pos, nn_vals_t)
            x_pos = 4 + g_idx*w/2
            ax1.bar(x_pos,
                    height=np.mean(ww_vals_t),
                    yerr=np.std(ww_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            scatter_datapoints(ax1, w, x_pos, ww_vals_t)

        # Trnsitions from REM
        # control
        x_pos = 0 - w/2
        ax2.bar(x_pos,
                height=np.mean(rn_vals_c),
                yerr=np.std(rn_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax2, w, x_pos, rn_vals_c)
        x_pos = 2 - w/2
        ax2.bar(x_pos,
                height=np.mean(rw_vals_c),
                yerr=np.std(rw_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax2, w, x_pos, rw_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            rw_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 0, 1]
            rn_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 0, 2]
            x_pos = 0 + g_idx*w/2
            ax2.bar(x_pos,
                    height=np.mean(rw_vals_t),
                    yerr=np.std(rw_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            scatter_datapoints(ax2, w, x_pos, rw_vals_t)
            x_pos = 2 + g_idx*w/2
            ax2.bar(x_pos,
                    height=np.mean(rn_vals_t),
                    yerr=np.std(rn_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            scatter_datapoints(ax2, w, x_pos, rn_vals_t)

        # Trnsitions from NREM
        # control
        x_pos = 0 - w/2
        ax3.bar(x_pos,
                height=np.mean(nr_vals_c),
                yerr=np.std(nr_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax3, w, x_pos, nr_vals_c)
        x_pos = 2 - w/2
        ax3.bar(x_pos,
                height=np.mean(nw_vals_c),
                yerr=np.std(nw_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax3, w, x_pos, nw_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            nr_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 2, 0]
            nw_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 2, 1]
            x_pos = 0 + g_idx*w/2
            ax3.bar(x_pos,
                    height=np.mean(nr_vals_t),
                    yerr=np.std(nr_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            scatter_datapoints(ax3, w, x_pos, nr_vals_t)
            x_pos = 2 + g_idx*w/2
            ax3.bar(x_pos,
                    height=np.mean(nw_vals_t),
                    yerr=np.std(nw_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            scatter_datapoints(ax3, w, x_pos, nw_vals_t)

        # Trnsitions from Wake
        # control
        x_pos = 0 - w/2
        ax4.bar(x_pos,
                height=np.mean(wr_vals_c),
                yerr=np.std(wr_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax4, w, x_pos, wr_vals_c)
        x_pos = 2 - w/2
        ax4.bar(x_pos,
                height=np.mean(wn_vals_c),
                yerr=np.std(wn_vals_c)/num_c,
                align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax4, w, x_pos, wn_vals_c)

        # tests.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            wr_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 1, 0]
            wn_vals_t = transmat_mat[bidx_group_list[g_idx]][:, 1, 2]
            x_pos = 0 + g_idx*w/2
            ax4.bar(x_pos,
                    height=np.mean(wr_vals_t),
                    yerr=np.std(wr_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            scatter_datapoints(ax4, w, x_pos, wr_vals_t)
            x_pos = 2 + g_idx*w/2
            ax4.bar(x_pos,
                    height=np.mean(wn_vals_t),
                    yerr=np.std(wn_vals_t)/num_t,
                    align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            scatter_datapoints(ax4, w, x_pos, wn_vals_t)
    else:
        # staying
        # single group
        x_pos = 0 - w/2
        ax1.bar(x_pos,
                height=np.mean(rr_vals_c),
                yerr=np.std(rr_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        scatter_datapoints(ax1, w, x_pos, rr_vals_c)
        x_pos = 2 - w/2
        ax1.bar(x_pos,
                height=np.mean(nn_vals_c),
                yerr=np.std(nn_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        scatter_datapoints(ax1, w, x_pos, nn_vals_c)
        x_pos = 4 - w/2
        ax1.bar(x_pos,
                height=np.mean(ww_vals_c),
                yerr=np.std(ww_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax1, w, x_pos, ww_vals_c)
        # Trnsitions from REM
        # single group
        x_pos = 0 - w/2
        ax2.bar(x_pos,
                height=np.mean(rn_vals_c),
                yerr=np.std(rn_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        scatter_datapoints(ax2, w, x_pos, rn_vals_c)
        x_pos = 2 - w/2
        ax2.bar(x_pos,
                height=np.mean(rw_vals_c),
                yerr=np.std(rw_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        scatter_datapoints(ax2, w, x_pos, rw_vals_c)
        # Trnsitions from NREM
        # single group
        x_pos = 0 - w/2
        ax3.bar(x_pos,
                height=np.mean(nr_vals_c),
                yerr=np.std(nr_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        scatter_datapoints(ax3, w, x_pos, nr_vals_c)
        x_pos = 2 - w/2
        ax3.bar(x_pos,
                height=np.mean(nw_vals_c),
                yerr=np.std(nw_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        scatter_datapoints(ax3, w, x_pos, nw_vals_c)
        # Trnsitions from Wake
        # single group
        x_pos = 0 - w/2
        ax4.bar(x_pos,
                height=np.mean(wr_vals_c),
                yerr=np.std(wr_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax4, w, x_pos, wr_vals_c)
        x_pos = 2 - w/2
        ax4.bar(x_pos,
                height=np.mean(wn_vals_c),
                yerr=np.std(wn_vals_c)/num_c,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax4, w, x_pos, wn_vals_c)

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
    filename = f'transition probability_barchart_{"_".join(mouse_groups_set)}'
    _savefig(output_dir, filename, fig)


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
    epoch_num = stagetime_stats['epoch_num']
    transmat_mat = np.array(stagetime_stats['transmat'])
    transmat_mat = np.vectorize(_odd)(transmat_mat, epoch_num)

    fig = _draw_transition_barchart(mouse_groups, transmat_mat)
    axes = fig.axes
    axes[0].set_ylabel('log odds to stay')
    axes[1].set_ylabel('log odds to transit from REM')
    axes[2].set_ylabel('log odds to transit from NREM')
    axes[3].set_ylabel('log odds to transit from Wake')
    fig.suptitle('transition probability (log odds)')
    filename = f'transition probability_barchart_logodds_{"_".join(mouse_groups_set)}'
    _savefig(output_dir, filename, fig)


def _draw_swtransition_barchart(mouse_groups, swtrans_mat):
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    w = 0.8  # bar width
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(['Psw', 'Pws'])

    if num_groups > 1:
        # control group (always index: 0)
        num_c = np.sum(bidx_group_list[0])
        sw_vals_c = swtrans_mat[bidx_group_list[0]][:, 0]
        ws_vals_c = swtrans_mat[bidx_group_list[0]][:, 1]

        ## Psw and Pws
        x_pos = 0 - w/2
        ax.bar(x_pos,
            height=np.mean(sw_vals_c),
            yerr=np.std(sw_vals_c)/num_c,
            align='center', width=w, capsize=6, color='gray', alpha=0.6)
        scatter_datapoints(ax, w, x_pos, sw_vals_c)
        x_pos = 2 - w/2
        ax.bar(x_pos,
            height=np.mean(ws_vals_c),
            yerr=np.std(ws_vals_c)/num_c,
            align='center', width=w, capsize=6, color='gray', alpha=0.6)
        scatter_datapoints(ax, w, x_pos, ws_vals_c)

        # test group index: g_idx.
        for g_idx in range(1, num_groups):
            num_t = np.sum(bidx_group_list[g_idx])
            sw_vals_t = swtrans_mat[bidx_group_list[g_idx]][:, 0]
            x_pos = 0 + g_idx*w/2
            ax.bar(x_pos,
                height=np.mean(sw_vals_t),
                yerr=np.std(sw_vals_t)/num_t,
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            scatter_datapoints(ax, w, x_pos, sw_vals_t)

            ws_vals_t = swtrans_mat[bidx_group_list[g_idx]][:, 1]
            x_pos = 2 + g_idx*w/2
            ax.bar(x_pos,
                height=np.mean(ws_vals_t),
                yerr=np.std(ws_vals_t)/num_t,
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            scatter_datapoints(ax, w, x_pos, ws_vals_t)
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
        scatter_datapoints(ax, w, x_pos, sw_vals)

        ws_vals = swtrans_mat[bidx_group_list[g_idx]][:, 1]
        x_pos = 2 + g_idx*w/2
        ax.bar(x_pos,
            height=np.mean(ws_vals),
            yerr=np.std(ws_vals)/num,
            align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax, w, x_pos, ws_vals)

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
    filename = f'sleep-wake transition probability_barchart_{"_".join(mouse_groups_set)}'
    _savefig(output_dir, filename, fig)


def draw_swtransition_barchart_logodds(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    swtrans_mat = np.array(stagetime_stats['swtrans'])
    swtrans_mat = np.vectorize(_odd)(swtrans_mat, epoch_num)

    fig = _draw_swtransition_barchart(mouse_groups, swtrans_mat)
    axes = fig.axes
    axes[0].set_ylabel('log odds to transit\n between sleep and wake')
    fig.suptitle('sleep/wake trantision probability (log odds)')
    filename = f'sleep-wake transition probability_barchart_logodds_{"_".join(mouse_groups_set)}'
    _savefig(output_dir, filename, fig)


def log_psd_inv(y, normalizing_fac, normalizing_mean):
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
    psd_mat = np.vectorize(log_psd_inv)(psd_norm_mat, nf, nm)

    return psd_mat


def make_target_psd_info(mouse_info_df, epoch_range, stage_ext):
    """makes PSD information sets for subsequent static analysis for each mouse:
    Arguments:
        mouse_info_df {pd.DataFram} -- an dataframe given by mouse_info_collected()
        sample_freq {int} -- sampling frequency
        epoch_range {slice} -- a range of target epochs
        stage_ext {str} -- a file sub-extention (e.g. 'faster2' for *.faster2.csv)

    Returns:
        psd_info_list [dict] --  A list of dict:
            'exp_label', 'mouse_group':mouse group, 'mouse_id':mouse id,
            'device_label', 'stage_call', 'bidx_rem', 'bidx_nrem','bidx_wake':bidx_wake, 
            'bidx_unknown', 'conv_psd':conv_psd'
    NOTE:
        len(conv_psd) <= (len(stage_call)=len(bidx_rem)=len(other bidx), because the
        unknown-stages' epochs do not have PSD. Therefore users should trim down the index
        arrays with bidx_unknown to extract elements from conv_psd:
        
        > bidx_rem_known = psd_info['bidx_rem'][~bidx_unknown]
        > psd_mean_rem  = np.apply_along_axis(np.mean, 0, conv_psd[bidx_rem_known, :])
    """    

    # target PSDs are from known-stage's, within-the-epoch-range, and good epochs
    psd_info_list = []
    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label'].strip()
        stats_report = r['Stats report'].strip().upper()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']

        if stats_report == 'NO':
            print_log(f'[{i+1}] Skipping: {faster_dir} {device_label}')
            continue
        print_log(f'[{i+1}] Reading PSD and stage of: {faster_dir} {device_label}')

        # read stage of the mouse
        stage_call, nan_eeg, outlier_eeg = et.read_stages_with_eeg_diagnosis(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)
        epoch_num = len(stage_call)
        
        # read the normalized EEG PSDs and the associated normalization factors and means
        # ==NOTE==  The unknown-stages' epochs do not have PSD (i.e. len(psd) <= epoch_num )
        pkl_path_eeg = os.path.join(
            faster_dir, 'result', 'PSD', f'{device_label}_EEG_PSD.pkl')
        with open(pkl_path_eeg, 'rb') as pkl:
            snorm_psd = pickle.load(pkl)

        # convert the spectrum normalized PSD to the conventional PSD
        conv_psd = conv_PSD_from_snorm_PSD(snorm_psd)

        # Break at the error: the unknown stage's PSD is not recoverable (even a manual annotator may want ...)
        bidx_unknown = snorm_psd['bidx_unknown']
        if not np.all(stage_call[bidx_unknown] == 'UNKNOWN'):
            print_log('[Error] "unknown" epoch is not recoverable. Check the consistency between the PSD and the stage files.')
            idx = list(np.where(bidx_unknown)[
                    0][stage_call[bidx_unknown] != 'UNKNOWN'])
            print_log(f'... in stage file at {idx}')
            break

        # good PSD should have the nan- and outlier-ratios of less than 1%
        bidx_good_psd = (nan_eeg < 0.01) & (outlier_eeg < 0.01)
        
        # bidx_target: bidx for the selected range
        bidx_selected = np.repeat(False, epoch_num)
        bidx_selected[epoch_range] = True
        bidx_target = bidx_selected & bidx_good_psd

        print_log(f'    Target epoch range: {epoch_range.start}-{epoch_range.stop} ({epoch_range.stop-epoch_range.start} epochs out of {epoch_num} epochs)\n'\
              f'    Unknown epochs in the range: {np.sum(bidx_unknown & bidx_selected)} ({100*np.sum(bidx_unknown & bidx_selected)/np.sum(bidx_selected):.3f}%)\n'\
              f'    Outlier or NA epochs in the range: {np.sum(~bidx_good_psd & bidx_selected)} ({100*np.sum(~bidx_good_psd & bidx_selected)/np.sum(bidx_selected):.3f}%)')

        bidx_rem = (stage_call == 'REM') & bidx_target
        bidx_nrem = (stage_call == 'NREM') & bidx_target
        bidx_wake = (stage_call == 'WAKE') & bidx_target 
        
        psd_info_list.append({'exp_label':exp_label,
                        'mouse_group':mouse_group,
                        'mouse_id':mouse_id,
                        'device_label':device_label,
                        'stage_call':stage_call,
                        'bidx_rem':bidx_rem, 
                        'bidx_nrem':bidx_nrem, 
                        'bidx_wake':bidx_wake, 
                        'bidx_unknown':bidx_unknown, 
                        'bidx_target': bidx_target,
                        'conv_psd':conv_psd})
        
    return psd_info_list


def make_psd_delta_timeseries_nrem_df(psd_info_list, sample_freq, epoch_num, epoch_range, summary_func=np.mean):
    # frequency bins
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)
    bidx_delta_freq = (freq_bins<4) # 11 bins

    # make the NREM delta-power timeseries
    psd_delta_timeseries_nrem_df = pd.DataFrame()
    for psd_info in psd_info_list:
        bidx_unknown = psd_info['bidx_unknown']
        stage_call = psd_info['stage_call']
        bidx_target = psd_info['bidx_target']
        bidx_nrem_known = psd_info['bidx_nrem'][~bidx_unknown]
        conv_psd = psd_info['conv_psd']
        psd_delta_timeseries_nrem = np.repeat(np.nan, epoch_num)
        psd_delta_timeseries_nrem[~bidx_unknown & (stage_call=='NREM') & bidx_target] = np.apply_along_axis(summary_func, 1, conv_psd[bidx_nrem_known, :][:,bidx_delta_freq])
        psd_delta_timeseries_nrem = psd_delta_timeseries_nrem[epoch_range] # extract epochs of the selected range
        psd_delta_timeseries_nrem_df = psd_delta_timeseries_nrem_df.append(
            [[psd_info['exp_label'], psd_info['mouse_group'], psd_info['mouse_id'], psd_info['device_label']] + psd_delta_timeseries_nrem.tolist()], ignore_index=True)

    epoch_columns = [f'epoch{x+1}' for x in np.arange(epoch_range.start, epoch_range.stop)]
    column_names = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label'] + epoch_columns
    psd_delta_timeseries_nrem_df.columns = column_names

    return psd_delta_timeseries_nrem_df


def make_psd_delta_timeseries_df(psd_info_list, sample_freq, epoch_num, epoch_range, summary_func=np.mean):
    # frequency bins
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)
    bidx_delta_freq = (freq_bins<4) # 11 bins

    # make the NREM delta-power timeseries
    psd_delta_timeseries_df = pd.DataFrame()
    for psd_info in psd_info_list:
        bidx_unknown = psd_info['bidx_unknown']
        bidx_target = psd_info['bidx_target']
        bidx_target_known = bidx_target[~bidx_unknown]
        conv_psd = psd_info['conv_psd']
        psd_delta_timeseries = np.repeat(np.nan, epoch_num)
        psd_delta_timeseries[~bidx_unknown & bidx_target] = np.apply_along_axis(summary_func, 1, conv_psd[bidx_target_known, :][:,bidx_delta_freq])
        psd_delta_timeseries = psd_delta_timeseries[epoch_range] # extract epochs of the selected range
        psd_delta_timeseries_df = psd_delta_timeseries_df.append(
            [[psd_info['exp_label'], psd_info['mouse_group'], psd_info['mouse_id'], psd_info['device_label']] + psd_delta_timeseries.tolist()], ignore_index=True)

    epoch_columns = [f'epoch{x+1}' for x in np.arange(epoch_range.start, epoch_range.stop)]
    column_names = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label'] + epoch_columns
    psd_delta_timeseries_df.columns = column_names

    return psd_delta_timeseries_df


def make_psd_profile(psd_info_list, sample_freq, epoch_range, stage_ext):
    """makes summary PSD statics of each mouse:
            psd_mean_df: summary (default: mean) of PSD profiles for each stage for each mice.

    Arguments:
        psd_info_list {[np.array]} -- a list of psd_info given by make_target_psd_info()
        sample_freq {int} -- sampling frequency
        epoch_range {slice} -- a range of target epochs
        stage_ext {str} -- a file sub-extention (e.g. 'faster2' for *.faster2.csv)

    Returns:
        psd_summary_df {pd.DataFrame} --  Experiment label, Mouse group, Mouse ID, 
            Device label, Stage, [freq_bins...]

    """

    # frequency bins
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    psd_summary_df = pd.DataFrame()
    for psd_info in psd_info_list:
        device_label = psd_info['device_label']
        mouse_group = psd_info['mouse_group']
        mouse_id = psd_info['mouse_id']
        exp_label = psd_info['exp_label']
        conv_psd = psd_info['conv_psd']

        bidx_unknown = psd_info['bidx_unknown']
        bidx_rem_known = psd_info['bidx_rem'][~bidx_unknown]
        bidx_nrem_known = psd_info['bidx_nrem'][~bidx_unknown]
        bidx_wake_known = psd_info['bidx_wake'][~bidx_unknown]
 
        psd_summary_rem  = np.apply_along_axis(np.mean, 0, conv_psd[bidx_rem_known, :])
        psd_summary_nrem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_nrem_known, :])
        psd_summary_wake = np.apply_along_axis(np.mean, 0, conv_psd[bidx_wake_known, :])

        psd_summary_df = psd_summary_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'REM'] + psd_summary_rem.tolist()], ignore_index=True)
        psd_summary_df = psd_summary_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'NREM'] + psd_summary_nrem.tolist()], ignore_index=True)
        psd_summary_df = psd_summary_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'Wake'] + psd_summary_wake.tolist()], ignore_index=True)

    freq_columns = [f'f@{x}' for x in freq_bins.tolist()]
    column_names = ['Experiment label', 'Mouse group',
                    'Mouse ID', 'Device label', 'Stage'] + freq_columns
    psd_summary_df.columns = column_names

    return psd_summary_df


def draw_PSDs_individual(psd_profiles_df, sample_freq, y_label, output_dir, opt_label=''):
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    # mouse_set
    mouse_list = psd_profiles_df['Mouse ID'].tolist()
    # unique elements with preseved order
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    # draw individual PSDs
    for m in mouse_set:
        fig = Figure(figsize=(16, 4))
        fig.subplots_adjust(wspace=0.25)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel(f'REM\n{y_label}')
        ax2.set_xlabel('freq. [Hz]')
        ax2.set_ylabel(f'NREM\n{y_label}')
        ax3.set_xlabel('freq. [Hz]')
        ax3.set_ylabel(f'Wake\n{y_label}')

        x = freq_bins
        df = psd_profiles_df.loc[psd_profiles_df['Mouse ID'] == m]
        y = df.loc[df['Stage'] == 'REM'].iloc[0].values[5:]
        ax1.plot(x, y, color=stage.COLOR_REM)

        y = df.loc[df['Stage'] == 'NREM'].iloc[0].values[5:]
        ax2.plot(x, y, color=stage.COLOR_NREM)

        y = df.loc[df['Stage'] == 'Wake'].iloc[0].values[5:]
        ax3.plot(x, y, color=stage.COLOR_WAKE)

        mouse_tag_list = [str(x) for x in df.iloc[0, 0:4]]
        fig.suptitle(
            f'Powerspectrum density: {"  ".join(mouse_tag_list)}')
        filename = f'PSD_{opt_label}I_{"_".join(mouse_tag_list)}'
        _savefig(output_dir, filename, fig)


def draw_PSDs_group(psd_profiles_df, sample_freq, y_label, output_dir, opt_label=''):
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    # mouse_group_set
    mouse_group_list = psd_profiles_df['Mouse group'].tolist()
    # unique elements with preseved order
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)

    # draw gropued PSD
    # _c of Control (assuming index = 0 is a control mouse)
    df = psd_profiles_df[psd_profiles_df['Mouse group'] == mouse_group_set[0]]

    psd_mean_mat_rem_c = df[df['Stage'] == 'REM'].iloc[:, 5:].values
    psd_mean_mat_nrem_c = df[df['Stage'] == 'NREM'].iloc[:, 5:].values
    psd_mean_mat_wake_c = df[df['Stage'] == 'Wake'].iloc[:, 5:].values
    num_c = psd_mean_mat_wake_c.shape[0]

    psd_mean_rem_c = np.apply_along_axis(np.mean, 0, psd_mean_mat_rem_c)
    psd_sem_rem_c = np.apply_along_axis(
        np.std, 0, psd_mean_mat_rem_c)/np.sqrt(num_c)
    psd_mean_nrem_c = np.apply_along_axis(np.mean, 0, psd_mean_mat_nrem_c)
    psd_sem_nrem_c = np.apply_along_axis(
        np.std, 0, psd_mean_mat_nrem_c)/np.sqrt(num_c)
    psd_mean_wake_c = np.apply_along_axis(np.mean, 0, psd_mean_mat_wake_c)
    psd_sem_wake_c = np.apply_along_axis(
        np.std, 0, psd_mean_mat_wake_c)/np.sqrt(num_c)

    x = freq_bins
    if len(mouse_group_set)>1:
        for g_idx in range(1, len(mouse_group_set)):
            fig = Figure(figsize=(16, 4))
            fig.subplots_adjust(wspace=0.25)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.set_xlabel('freq. [Hz]')
            ax1.set_ylabel(f'REM\n{y_label}')
            ax2.set_xlabel('freq. [Hz]')
            ax2.set_ylabel(f'NREM\n{y_label}')
            ax3.set_xlabel('freq. [Hz]')
            ax3.set_ylabel(f'Wake\n{y_label}')

            # _t of Treatment
            df = psd_profiles_df[psd_profiles_df['Mouse group']
                                == mouse_group_set[g_idx]]
            psd_mean_mat_rem_t = df[df['Stage'] == 'REM'].iloc[:, 5:].values
            psd_mean_mat_nrem_t = df[df['Stage'] == 'NREM'].iloc[:, 5:].values
            psd_mean_mat_wake_t = df[df['Stage'] == 'Wake'].iloc[:, 5:].values
            num_t = psd_mean_mat_wake_t.shape[0]

            psd_mean_rem_t = np.apply_along_axis(np.mean, 0, psd_mean_mat_rem_t)
            psd_sem_rem_t = np.apply_along_axis(
                np.std, 0, psd_mean_mat_rem_t)/np.sqrt(num_t)
            psd_mean_nrem_t = np.apply_along_axis(np.mean, 0, psd_mean_mat_nrem_t)
            psd_sem_nrem_t = np.apply_along_axis(
                np.std, 0, psd_mean_mat_nrem_t)/np.sqrt(num_t)
            psd_mean_wake_t = np.apply_along_axis(np.mean, 0, psd_mean_mat_wake_t)
            psd_sem_wake_t = np.apply_along_axis(
                np.std, 0, psd_mean_mat_wake_t)/np.sqrt(num_t)

            y = psd_mean_rem_c
            y_sem = psd_sem_rem_c
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                            y + y_sem, color='grey', alpha=0.3)

            y = psd_mean_rem_t
            y_sem = psd_sem_rem_t
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_REM, alpha=0.3)

            y = psd_mean_nrem_c
            y_sem = psd_sem_nrem_c
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                            y + y_sem, color='grey', alpha=0.3)

            y = psd_mean_nrem_t
            y_sem = psd_sem_nrem_t
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            y = psd_mean_wake_c
            y_sem = psd_sem_wake_c
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem,
                            y + y_sem, color='grey', alpha=0.3)

            y = psd_mean_wake_t
            y_sem = psd_sem_wake_t
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(
                f'Powerspectrum density: {mouse_group_set[0]} (n={num_c}) v.s. {mouse_group_set[g_idx]} (n={num_t})')
            filename = f'PSD_{opt_label}G_{mouse_group_set[0]}_vs_{mouse_group_set[g_idx]}'
            _savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0
        fig = Figure(figsize=(16, 4))
        fig.subplots_adjust(wspace=0.25)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel(f'REM\n{y_label}')
        ax2.set_xlabel('freq. [Hz]')
        ax2.set_ylabel(f'NREM\n{y_label}')
        ax3.set_xlabel('freq. [Hz]')
        ax3.set_ylabel(f'Wake\n{y_label}')

        # _t of Treatment
        df = psd_profiles_df[psd_profiles_df['Mouse group']
                            == mouse_group_set[g_idx]]
        psd_mean_mat_rem_t = df[df['Stage'] == 'REM'].iloc[:, 5:].values
        psd_mean_mat_nrem_t = df[df['Stage'] == 'NREM'].iloc[:, 5:].values
        psd_mean_mat_wake_t = df[df['Stage'] == 'Wake'].iloc[:, 5:].values
        num_t = psd_mean_mat_wake_t.shape[0]

        psd_mean_rem_t = np.apply_along_axis(np.mean, 0, psd_mean_mat_rem_t)
        psd_sem_rem_t = np.apply_along_axis(
            np.std, 0, psd_mean_mat_rem_t)/np.sqrt(num_t)
        psd_mean_nrem_t = np.apply_along_axis(np.mean, 0, psd_mean_mat_nrem_t)
        psd_sem_nrem_t = np.apply_along_axis(
            np.std, 0, psd_mean_mat_nrem_t)/np.sqrt(num_t)
        psd_mean_wake_t = np.apply_along_axis(np.mean, 0, psd_mean_mat_wake_t)
        psd_sem_wake_t = np.apply_along_axis(
            np.std, 0, psd_mean_mat_wake_t)/np.sqrt(num_t)

        y = psd_mean_rem_t
        y_sem = psd_sem_rem_t
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_REM, alpha=0.3)

        y = psd_mean_nrem_t
        y_sem = psd_sem_nrem_t
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

        y = psd_mean_wake_t
        y_sem = psd_sem_wake_t
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

        fig.suptitle(
            f'Powerspectrum density: {mouse_group_set[g_idx]} (n={num_t})')
        filename = f'PSD_{opt_label}G_{mouse_group_set[g_idx]}'
        _savefig(output_dir, filename, fig)


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

    sleep_stats_df = sleep_stats_df.append([row1, row2, row3])
    for i, bidx in enumerate(bidx_group_list[1:]):
        idx = i+1
        mg = mouse_groups_set[idx]
        bidx = bidx_group_list[idx]
        num = np.sum(bidx)
        rem_values_t = stagetime_df['REM'].values[bidx]
        nrem_values_t = stagetime_df['NREM'].values[bidx]
        wake_values_t = stagetime_df['Wake'].values[bidx]

        tr = test_two_sample(rem_values_c,  rem_values_t)  # test for REM
        tn = test_two_sample(nrem_values_c, nrem_values_t)  # test for NREM
        tw = test_two_sample(wake_values_c, wake_values_t)  # test for Wake
        row1 = [mg, 'REM',  num, np.mean(rem_values_t),  np.std(
            rem_values_t),  tr['p_value'], tr['stars'], tr['method']]
        row2 = [mg, 'NREM', num, np.mean(nrem_values_t), np.std(
            nrem_values_t), tn['p_value'], tn['stars'], tn['method']]
        row3 = [mg, 'Wake', num, np.mean(wake_values_t), np.std(
            wake_values_t), tw['p_value'], tw['stars'], tw['method']]

        sleep_stats_df = sleep_stats_df.append([row1, row2, row3])

    sleep_stats_df.columns = ['Mouse group', 'Stage type',
                              'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    stagetime_df = stagetime_df.round(
        {'REM': 2, 'NREM': 2, 'Wake': 2, 'Unknown': 2})

    sleep_stats_df.to_csv(os.path.join(
        output_dir, 'stage-time_stats_table.csv'), index=False)
    stagetime_df.to_csv(os.path.join(
        output_dir, 'stage-time_table.csv'), index=False)


def make_psd_domain(psd_profiles_df, summary_func=np.mean):
    """ makes PSD power averaged within frequency domains of each stage for each mice

    Arguments:
        psd_profiles_df {pd.DataFrame} -- a dataframe given by make_psd_profile()
        summary_func {function} -- a function for summarizing PSD

    Returns:
        [pd.DataFrame] -- Experiment label, Mouse group, Mouse ID, 
            Device label, Stage, Slow, Delta w/o slow, Delta, Theta
    """
    # get freq_bins from column names
    freq_bin_columns = psd_profiles_df.columns[5:].tolist()
    freq_bins = np.array([float(x.strip().split('@')[1])
                          for x in freq_bin_columns])

    # frequency domains
    bidx_theta_freq = (freq_bins >= 4) & (freq_bins < 10)  # 15 bins
    bidx_delta_freq = (freq_bins < 4)  # 11 bins
    bidx_delta_wo_slow_freq = (1 <= freq_bins) & (
        freq_bins < 4)  # 8 bins (delta without slow)
    bidx_slow_freq = (freq_bins < 1)  # 3 bins

    # make psd_domain_df
    row_list = []
    for _, r in psd_profiles_df.iterrows():
        infos = r[:5]
        powers = r[5:]
        powers_slow = powers[bidx_slow_freq]
        powers_delta_wo_slow = powers[bidx_delta_wo_slow_freq]
        powers_delta = powers[bidx_delta_freq]
        powers_theta = powers[bidx_theta_freq]

        slow_p = summary_func(powers_slow)
        delta_wo_slow_p = summary_func(powers_delta_wo_slow)
        delta_p = summary_func(powers_delta)
        theta_p = summary_func(powers_theta)
        domain_powers = pd.Series(
            [slow_p, delta_wo_slow_p, delta_p, theta_p], index=DOMAIN_NAMES)

        row = pd.concat([infos, domain_powers])
        row_list.append(row)

    psd_domain_df = pd.DataFrame(row_list)
    return psd_domain_df


def make_psd_stats(psd_domain_df):
    """ makes a table of statistical tests for each frequency domains between groups 

    Arguments:
        psd_domain_df {pd.DataFrame} -- a dataframe given by make_psd_domain()

    Returns:
        [pd.DataFrame] -- # mouse_group, stage_type, wave_type, num, 
                            mean, SD, pvalue, star, method
    """
    def _domain_powers_by_group(psd_domain_df, group):
        bidx_group = (psd_domain_df['Mouse group'] == group)
        bidx_rem = (psd_domain_df['Stage'] == 'REM')
        bidx_nrem = (psd_domain_df['Stage'] == 'NREM')
        bidx_wake = (psd_domain_df['Stage'] == 'Wake')

        domain_powers_rem = psd_domain_df.loc[bidx_group &
                                              bidx_rem][DOMAIN_NAMES]
        domain_powers_nrem = psd_domain_df.loc[bidx_group &
                                               bidx_nrem][DOMAIN_NAMES]
        domain_powers_wake = psd_domain_df.loc[bidx_group &
                                               bidx_wake][DOMAIN_NAMES]

        return [domain_powers_rem, domain_powers_nrem, domain_powers_wake]

    psd_stats_df = pd.DataFrame()
    # mouse_group_set
    mouse_group_list = psd_domain_df['Mouse group'].tolist()
    # unique elements with preseved order
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)

    # control
    group_c = mouse_group_set[0]  # index=0 should be always control group

    # There are 3 powers_domains_[stages] where [stages] are [REM, NREM, Wake].
    # Each powers_domains_[stage] contains 4 Series of domain powers:
    # [slow x mice, delta wo slow x mice, delta x mice, theta x mice]
    # Therefore, the following loop results in 12 rows.
    stage_names = ['REM', 'NREM', 'Wake']
    powers_domains_stages_c = _domain_powers_by_group(psd_domain_df, group_c)

    rows = []
    for stage_name, powers_domains in zip(stage_names, powers_domains_stages_c):
        for domain_name in DOMAIN_NAMES:
            powers = powers_domains[domain_name]
            num = len(powers)
            rows.append([group_c, stage_name, domain_name, num,
                         np.mean(powers),  np.std(powers), np.nan, None, None])

    psd_stats_df = psd_stats_df.append(rows)

    # treatment
    for group_t in mouse_group_set[1:]:
        rows = []
        powers_domains_stages_t = _domain_powers_by_group(
            psd_domain_df, group_t)
        for stage_name, powers_domains_c, powers_domains_t in zip(stage_names, powers_domains_stages_c, powers_domains_stages_t):
            for domain_name in DOMAIN_NAMES:
                powers_c = powers_domains_c[domain_name]
                powers_t = powers_domains_t[domain_name]
                test = test_two_sample(powers_c, powers_t)
                num = len(powers_t)
                rows.append([group_t, stage_name, domain_name, num,
                             np.mean(powers_t),  np.std(powers_t), test['p_value'], test['stars'], test['method']])

        psd_stats_df = psd_stats_df.append(rows)

    psd_stats_df.columns = ['Mouse group', 'Stage type',
                            'Wake type', 'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    return psd_stats_df


def write_psd_stats(psd_profiles_df, output_dir, opt_label='', summary_func=np.mean):
    """ writes three PSD tables:
        1. psd_profile.csv: mean PSD profile of each stage for each mice
        2. psd_freq_domain_table.csv: PSD power averaged within frequency domains of each stage for each mice 
        3. psd_stats_table.csv: statistical tests for each frequency domains between groups 
    """

    psd_domain_df = make_psd_domain(psd_profiles_df, summary_func)
    psd_stats_df = make_psd_stats(psd_domain_df)

    # write tabels
    psd_profiles_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile.csv'), index=False)
    psd_domain_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}freq_domain_table.csv'), index=False)
    psd_stats_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}stats_table.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--faster2_dirs", required=True, nargs="*",
                        help="paths to the FASTER2 directories")
    parser.add_argument("-e", "--epoch_range",
                        help="a range of epochs to be summaried (default: '0:epoch_num'")
    parser.add_argument("-s", "--stage_ext",
                        help="the sub-extention of the stage file (default: faster2)")
    parser.add_argument("-o", "--output_dir",
                        help="a path to the output files (default: the first FASTER2 directory)")

    args = parser.parse_args()

    faster_dir_list = [os.path.abspath(x) for x in args.faster2_dirs]
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = None
    stage_ext = args.stage_ext

    # collect mouse_infos of the specified (multiple) FASTER dirs
    mouse_info_collected = collect_mouse_info_df(faster_dir_list)
    mouse_info_df = mouse_info_collected['mouse_info']
    epoch_num = mouse_info_collected['epoch_num']
    sample_freq = mouse_info_collected['sample_freq']

    # set the epoch range to be summarized
    if args.epoch_range:
        # use the range given by the command line option
        e_range = [
            int(x.strip()) if x else None for x in args.epoch_range.split(':')]
        epoch_range = slice(*e_range)
        epoch_num = e_range[1] - e_range[0]
    else:
        # default: use the all epochs
        epoch_range = slice(0, epoch_num, None)

    # set the file sub-extention of the stage files to be summarized
    if stage_ext == None:
        # default: 'faster2' for *.faster2.csv
        stage_ext = 'faster2'

    # set the output directory
    if output_dir == None:
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

    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(output_dir, 'log', f'summary.{dt_str}.log'))

    # prepare stagetime statistics
    stagetime_stats = make_summary_stats(mouse_info_df, epoch_range, stage_ext)

    # write a table of stats
    write_sleep_stats(stagetime_stats, output_dir)

    # draw stagetime profile of individual mice
    draw_stagetime_profile_individual(stagetime_stats, output_dir)

    # draw stagetime profile of grouped mice
    draw_stagetime_profile_grouped(stagetime_stats, output_dir)

    # draw stagetime circadian profile of individual mice
    draw_stagetime_circadian_profile_indiviudal(stagetime_stats, output_dir)

    # draw stagetime circadian profile of groups
    draw_stagetime_circadian_profile_grouped(stagetime_stats, output_dir)

    # draw stagetime barchart
    draw_stagetime_barchart(stagetime_stats, output_dir)

    # draw stagetime profile of individual mice
    draw_swtrans_profile_individual(stagetime_stats, output_dir)

    # draw stagetime profile of grouped mice
    draw_swtrans_profile_grouped(stagetime_stats, output_dir)

    # draw stagetime profile of individual mice
    draw_swtrans_circadian_profile_individual(stagetime_stats, output_dir)

    # draw stagetime profile of individual mice
    draw_swtrans_circadian_profile_grouped(stagetime_stats, output_dir)

    # draw transition barchart (probability)
    draw_transition_barchart_prob(stagetime_stats, output_dir)

    # draw transition barchart (log odds)
    draw_transition_barchart_logodds(stagetime_stats, output_dir)

    # draw sleep/wake transition probability
    draw_swtransition_barchart_prob(stagetime_stats, output_dir)

    # draw sleep/wake transition probability (log odds)
    draw_swtransition_barchart_logodds(stagetime_stats, output_dir)

    # prepare Powerspectrum density (PSD) profiles for individual mice
    # list of {simple exp info, target info, psd (epoch_num, 129)} for each mouse
    psd_info_list = make_target_psd_info(mouse_info_df, epoch_range, stage_ext)
 
    # log version of psd_info
    print_log('Making the log version of the PSD information')
    log_psd_info_list = copy.deepcopy(psd_info_list)
    for log_psd_info in log_psd_info_list:
        log_psd_info['conv_psd'] = 10*np.log10(log_psd_info['conv_psd'])

    # percentage version of psd_info
    print_log('Making the percentage version of the PSD information')
    percentage_psd_info_list = copy.deepcopy(psd_info_list)
    for percentage_psd_info in percentage_psd_info_list:
        conv_psd = percentage_psd_info['conv_psd']
        percentage_psd_mat = np.zeros(conv_psd.shape)
        for i, p in enumerate(conv_psd): # row wise
            percentage_psd_mat[i,:] = 100*p / np.sum(p)
        percentage_psd_info['conv_psd'] = percentage_psd_mat

    # PSD profiles
    psd_profiles_df = make_psd_profile(
        psd_info_list, sample_freq, epoch_range, stage_ext)
    log_psd_profiles_df = make_psd_profile(
        log_psd_info_list, sample_freq, epoch_range, stage_ext)
    percentage_psd_profiles_df = make_psd_profile(
        percentage_psd_info_list, sample_freq, epoch_range, stage_ext)

    # write a table of PSD
    write_psd_stats(psd_profiles_df, output_dir)
    write_psd_stats(log_psd_profiles_df, output_dir, 'log-')
    write_psd_stats(percentage_psd_profiles_df, output_dir, 'percentage-', np.sum)

    # draw PSDs
    print_log('Drawing the PSDs')
    draw_PSDs_individual(psd_profiles_df, sample_freq,
                         'normalized PSD [AU]', output_dir)
    draw_PSDs_individual(log_psd_profiles_df, sample_freq,
                         'normalized log PSD [AU]', output_dir, 'log-')
    draw_PSDs_individual(percentage_psd_profiles_df, sample_freq,
                         'normalized log PSD [AU]', output_dir, 'percentage-')

    draw_PSDs_group(psd_profiles_df, sample_freq,
                    'normalized PSD [AU]', output_dir)
    draw_PSDs_group(log_psd_profiles_df, sample_freq,
                    'normalized log PSD [AU]', output_dir, 'log-')
    draw_PSDs_group(percentage_psd_profiles_df, sample_freq,
                    'normalized percentage PSD [%]', output_dir, 'percentage-')

    print_log('Making the delta-power timeseries')
    psd_delta_timeseries_df = make_psd_delta_timeseries_df(psd_info_list, sample_freq, epoch_num, epoch_range)
    print_log('Making the delta-power timeseries in NREM')
    psd_delta_timeseries_nrem_df = make_psd_delta_timeseries_nrem_df(psd_info_list, sample_freq, epoch_num, epoch_range)
    print_log('Making the delta-power timeseries (percentage)')
    percentage_psd_delta_timeseries_df = make_psd_delta_timeseries_df(percentage_psd_info_list, sample_freq, epoch_num, epoch_range, np.sum)
    print_log('Making the delta-power timeseries in NREM (percentage)')
    percentage_psd_delta_timeseries_nrem_df = make_psd_delta_timeseries_nrem_df(percentage_psd_info_list, sample_freq, epoch_num, epoch_range, np.sum)

    # draw delta-power timeseries
    print_log('Drawing the delta-power timeseries')
    draw_psd_delta_timeseries_individual(psd_delta_timeseries_df, 'Hourly delta power [AU]', output_dir)
    draw_psd_delta_timeseries_grouped(psd_delta_timeseries_df, 'Hourly delta power [AU]', output_dir)
    draw_psd_delta_timeseries_individual(percentage_psd_delta_timeseries_df, 'Hourly delta power [%]', output_dir, 'percentage_')
    draw_psd_delta_timeseries_grouped(percentage_psd_delta_timeseries_df, 'Hourly delta power [%]', output_dir, 'percentage_')
    draw_psd_delta_timeseries_individual(psd_delta_timeseries_nrem_df, 'Hourly NREM delta power [AU]', output_dir, 'NREM_')
    draw_psd_delta_timeseries_grouped(psd_delta_timeseries_nrem_df, 'Hourly NREM delta power [AU]', output_dir, 'NREM_')
    draw_psd_delta_timeseries_individual(percentage_psd_delta_timeseries_nrem_df, 'Hourly NREM delta power [%]', output_dir, 'NREM_percentage_')
    draw_psd_delta_timeseries_grouped(percentage_psd_delta_timeseries_nrem_df, 'Hourly NREM delta power [%]', output_dir, 'NREM_percentage_')
