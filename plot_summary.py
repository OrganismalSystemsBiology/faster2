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
    return ({'mouse_info':mouse_info_df, 'epoch_num':epoch_num, 'sample_freq':sample_freq})


def make_summary_stats(mouse_info_df, epoch_num):
    """ make summary statics of each mouse:
            stagetime in a day: how many minuites of stages each mouse spent in a day
            stage time profile: hourly profiles of stages over the recording
            stage circadian profile: hourly profiles of stages over a day
            transition matrix: transition probability matrix among each stage
            sw transitino: Sleep (NREM+REM) and Wake transition probability 
    
    Arguments:
        mouse_info_df {pd.DataFram} -- [description]
        epoch_num {int} -- [description]

    Returns:
        {'stagetime': pd.DataFrame, 
        'stagetime_profile': np.array(epoch_num), 
        'stagetime_circadian': np.array(24), 
        'transmat': np.array(3,3),
        'swtrans': np.array(2)} -- A dict of dataframe and arrays of summary stats  
    """
    stagetime_df = pd.DataFrame()
    stagetime_profile_list = []
    stagetime_circadian_profile_list = []
    transmat_list = []
    swtrans_list = []

    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label']
        mouse_group = r['Mouse group']
        mouse_id = r['Mouse ID']
        stats_report = r['Stats report'].strip().upper()
        exp_label = r['Experiment label']
        faster_dir = r['FASTER_DIR']
        if stats_report=='NO':
            print(f'[{i+1}] skipping stage: {faster_dir} {device_label}')
            continue
        
        print(f'[{i+1}] reading stage: {faster_dir} {device_label}')
        stage_call = et.read_stages(os.path.join(faster_dir, 'result'), device_label, 'faster2')
        stage_call = stage_call[:epoch_num]
        
        # stage time in a day
        rem, nrem, wake, unknown = stagetime_in_a_day(stage_call)
        stagetime_df = stagetime_df.append([[exp_label, mouse_group, mouse_id, device_label, rem, nrem, wake, unknown, stats_report]], ignore_index=True)
        
        # stage time profile
        stagetime_profile_list.append(stagetime_profile(stage_call))
        
        # stage circadian profile
        stagetime_circadian_profile_list.append(stagetime_circadian_profile(stage_call))
        
        # transition matrix
        transmat_list.append(transmat_from_stages(stage_call))

        # sw transition
        swtrans_list.append(swtrans_from_stages(stage_call))
        
    stagetime_df.columns = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label', 'REM', 'NREM', 'Wake', 'Unknown', 'Stats report']
    
    return({'stagetime': stagetime_df,
            'stagetime_profile': stagetime_profile_list,
            'stagetime_circadian': stagetime_circadian_profile_list,
            'transmat': transmat_list,
            'swtrans': swtrans_list})


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

    rem = 1440*np.sum(stage_call=="REM")/ndata
    nrem = 1440*np.sum(stage_call=="NREM")/ndata
    wake = 1440*np.sum(stage_call=="WAKE")/ndata
    unknown = 1440*np.sum(stage_call=="UNKNOWN")/ndata
    
    return (rem ,nrem, wake, unknown)


def stagetime_profile (stage_call):
    """ hourly profiles of stages over the recording
    
    Arguments:
        stage_call {np.array} -- an array of stage calls (e.g. ['WAKE', 
        'NREM', ...])
    
    Returns:
        [np.array(3, len(stage_calls))] -- each row corrensponds the 
        hourly profiles of stages over the recording (rem, nrem, wake)
    """
    sm = stage_call.reshape(-1, int(3600/stage.EPOCH_LEN_SEC)) # 60 min(3600 sec) bin
    rem = np.array([np.sum(s=='REM')*stage.EPOCH_LEN_SEC/60 for s in sm]) # unit minuite
    nrem = np.array([np.sum(s=='NREM')*stage.EPOCH_LEN_SEC/60 for s in sm]) # unit minuite
    wake = np.array([np.sum(s=='WAKE')*stage.EPOCH_LEN_SEC/60 for s in sm]) # unit minuite
    
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


def transmat_from_stages(stages):
    """transition probability matrix among each stage
    
    Arguments:
        stages {np.array} -- an array of stage calls (e.g. ['WAKE', 
        'NREM', ...])
    
    Returns:
        [np.array(3,3)] -- a 3x3 matrix of transition probabilites.
        Notice the order is REM, WAKE, NREM.
    """
    rr = np.sum((stages[:-1]=='REM') & (stages[1:]=='REM'))  # REM -> REM
    rw = np.sum((stages[:-1]=='REM') & (stages[1:]=='WAKE')) # REM -> Wake
    rn = np.sum((stages[:-1]=='REM') & (stages[1:]=='NREM')) # REM -> NREM

    wr = np.sum((stages[:-1]=='WAKE') & (stages[1:]=='REM'))  # Wake -> REM
    ww = np.sum((stages[:-1]=='WAKE') & (stages[1:]=='WAKE')) # Wake-> Wake
    wn = np.sum((stages[:-1]=='WAKE') & (stages[1:]=='NREM')) # Wake-> NREM

    nr = np.sum((stages[:-1]=='NREM') & (stages[1:]=='REM'))  # NREM -> REM
    nw = np.sum((stages[:-1]=='NREM') & (stages[1:]=='WAKE')) # NREM -> Wake
    nn = np.sum((stages[:-1]=='NREM') & (stages[1:]=='NREM')) # NREM -> NREM
 
    r_trans = rr + rw + rn
    w_trans = wr + ww + wn
    n_trans = nr + nw + nn
    transmat = np.array([[rr, rw, rn]/r_trans, [wr, ww, wn]/w_trans, [nr, nw, nn]/n_trans])
    
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
    rr = np.sum((stages[:-1]=='REM') & (stages[1:]=='REM'))  # REM -> REM
    rw = np.sum((stages[:-1]=='REM') & (stages[1:]=='WAKE')) # REM -> Wake
    rn = np.sum((stages[:-1]=='REM') & (stages[1:]=='NREM')) # REM -> NREM

    wr = np.sum((stages[:-1]=='WAKE') & (stages[1:]=='REM'))  # Wake -> REM
    ww = np.sum((stages[:-1]=='WAKE') & (stages[1:]=='WAKE')) # Wake-> Wake
    wn = np.sum((stages[:-1]=='WAKE') & (stages[1:]=='NREM')) # Wake-> NREM

    nr = np.sum((stages[:-1]=='NREM') & (stages[1:]=='REM')) # NREM -> REM
    nw = np.sum((stages[:-1]=='NREM') & (stages[1:]=='WAKE')) # NREM -> Wake
    nn = np.sum((stages[:-1]=='NREM') & (stages[1:]=='NREM')) # NREM -> NREM
 
    s_trans = rr + rw + rn + nr + nw + nn
    w_trans = wr + ww + wn
    swtrans = np.array([(rw+nw)/s_trans, (wn+wr)/w_trans]) # Psw, Pws
    
    return swtrans


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


def _set_common_features_stagetime_profile_rem(ax, x_max):
    r = 4 # a scale factor for y-axis
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


def draw_stagetime_profile_individual(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_profile_list = stagetime_stats['stagetime_profile']
    x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
    x = np.arange(x_max)
    for i, profile in enumerate(stagetime_profile_list):
        fig = Figure(figsize=(13,6))
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


        ax1.plot(x, profile[0,:], color=stage.COLOR_REM)
        ax2.plot(x, profile[1,:], color=stage.COLOR_NREM)
        ax3.plot(x, profile[2,:], color=stage.COLOR_WAKE)

        fig.suptitle(f'Stage-time profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'stage-time_profile_{"_".join(stagetime_df.iloc[i,0:4].values)}.jpg'
        fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


def draw_stagetime_profile_grouped(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_profile_list = stagetime_stats['stagetime_profile']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups==g for g in mouse_groups_set]

    # make stats of stagetime profile: mean and sd over each group
    stagetime_profile_mat = np.array(stagetime_profile_list) # REM, NREM, Wake
    stagetime_profile_stats_list = []
    for bidx in bidx_group_list:
        stagetime_profile_mean = np.apply_along_axis(np.mean, 0, stagetime_profile_mat[bidx])
        stagetime_profile_sd   = np.apply_along_axis(np.std, 0, stagetime_profile_mat[bidx])
        stagetime_profile_stats_list.append(np.array([stagetime_profile_mean, stagetime_profile_sd]))

    x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
    x = np.arange(x_max)
    if len(mouse_groups_set)>1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
            num = np.sum(bidx_group_list[g_idx])
            fig = Figure(figsize=(13,6))
            ax1 = fig.add_subplot(311, xmargin=0, ymargin=0)
            ax2 = fig.add_subplot(312, xmargin=0, ymargin=0)
            ax3 = fig.add_subplot(313, xmargin=0, ymargin=0)

            _set_common_features_stagetime_profile_rem(ax1, x_max)
            _set_common_features_stagetime_profile(ax2, x_max)
            _set_common_features_stagetime_profile(ax3, x_max)

            ## Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # REM
            y     = stagetime_profile_stats_list[0][0, 0, :]
            y_sem = stagetime_profile_stats_list[0][1, 0, :]/np.sqrt(num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                                y + y_sem, color='grey', alpha=0.3)
            ax1.set_ylabel('Hourly REM\n duration (min)')

            # NREM
            y     = stagetime_profile_stats_list[0][0, 1, :]
            y_sem = stagetime_profile_stats_list[0][1, 1, :]/np.sqrt(num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                                y + y_sem, color='grey', alpha=0.3)
            ax2.set_ylabel('Hourly NREM\n duration (min)')

            # Wake
            y     = stagetime_profile_stats_list[0][0, 2, :]
            y_sem = stagetime_profile_stats_list[0][1, 2, :]/np.sqrt(num_c)
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem/np.sqrt(num),
                                y + y_sem/np.sqrt(num), color='grey', alpha=0.3)
            ax3.set_ylabel('Hourly wake\n duration (min)')
            ax3.set_xlabel('Time (hours)')


            ## Treatment
            g_idx = 1
            num = np.sum(bidx_group_list[g_idx])
            # REM
            y     = stagetime_profile_stats_list[g_idx][0, 0, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                                y + y_sem, color=stage.COLOR_REM, alpha=0.3)

            # NREM
            y     = stagetime_profile_stats_list[g_idx][0, 1, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                                y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            # Wake
            y     = stagetime_profile_stats_list[g_idx][0, 2, :]
            y_sem = stagetime_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                                y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'stage-time_profile_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}.jpg'
            fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)
    else:
        # single group
        g_idx = 0

        num = np.sum(bidx_group_list[g_idx])
        x_max = epoch_num*stage.EPOCH_LEN_SEC/3600
        x = np.arange(x_max)
        fig = Figure(figsize=(13,6))
        ax1 = fig.add_subplot(311, xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(312, xmargin=0, ymargin=0)
        ax3 = fig.add_subplot(313, xmargin=0, ymargin=0)

        _set_common_features_stagetime_profile_rem(ax1, x_max)
        _set_common_features_stagetime_profile(ax2, x_max)
        _set_common_features_stagetime_profile(ax3, x_max)

        # REM
        y     = stagetime_profile_stats_list[g_idx][0, 0, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_REM, alpha=0.3)
        ax1.set_ylabel('Hourly REM\n duration (min)')

        # NREM
        y     = stagetime_profile_stats_list[g_idx][0, 1, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax2.set_ylabel('Hourly NREM\n duration (min)')

        # Wake
        y     = stagetime_profile_stats_list[g_idx][0, 2, :]
        y_sem = stagetime_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem/np.sqrt(num),
                            y + y_sem/np.sqrt(num), color=stage.COLOR_WAKE, alpha=0.3)
        ax3.set_ylabel('Hourly wake\n duration (min)')
        ax3.set_xlabel('Time (hours)')

        fig.suptitle(f'{mouse_groups_set[g_idx]} (n={num})')
        filename = f'stage-time_profile_{mouse_groups_set[g_idx]}.jpg'
        fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


def draw_stagetime_circadian_profile_indiviudal(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_circadian_list = stagetime_stats['stagetime_circadian']
    for i, circadian in enumerate(stagetime_circadian_list):
        x_max = 24
        x = np.arange(x_max)
        fig = Figure(figsize=(13,4))
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
        y    = circadian[0, 0, :]
        y_sem = circadian[1, 0, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_REM, alpha=0.3)

        # NREM
        y    = circadian[0, 1, :]
        y_sem = circadian[1, 1, :]/np.sqrt(num)
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

        # Wake
        y    = circadian[0, 2, :]
        y_sem = circadian[1, 2, :]/np.sqrt(num)
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

        fig.suptitle(f'Circadian stage-time profile: {"  ".join(stagetime_df.iloc[i,0:4].values)}')
        filename = f'stage-time_circadian_profile_{"_".join(stagetime_df.iloc[i,0:4].values)}.jpg'
        fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


def draw_stagetime_circadian_profile_grouped(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    stagetime_circadian_profile_list = stagetime_stats['stagetime_circadian']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups==g for g in mouse_groups_set]

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
    if len(mouse_groups_set)>1:
        for g_idx in range(1, len(mouse_groups_set)):        
            fig = Figure(figsize=(13,4))
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


            ## Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            # REM
            y     = stagetime_circadian_profile_stats_list[0][0, 0, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 0, :]/np.sqrt(num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                                y + y_sem, color='grey', alpha=0.3)

            # NREM
            y    = stagetime_circadian_profile_stats_list[0][0, 1, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 1, :]/np.sqrt(num_c)
            ax2.plot(x, y, color='grey')
            ax2.fill_between(x, y - y_sem,
                                y + y_sem, color='grey', alpha=0.3)

            # Wake
            y     = stagetime_circadian_profile_stats_list[0][0, 2, :]
            y_sem = stagetime_circadian_profile_stats_list[0][1, 2, :]/np.sqrt(num_c)
            ax3.plot(x, y, color='grey')
            ax3.fill_between(x, y - y_sem,
                                y + y_sem, color='grey', alpha=0.3)

            ## Treatment
            num = np.sum(bidx_group_list[g_idx])
            # REM
            y    = stagetime_circadian_profile_stats_list[g_idx][0, 0, :]
            y_sem= stagetime_circadian_profile_stats_list[g_idx][1, 0, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_REM)
            ax1.fill_between(x, y - y_sem,
                                y + y_sem, color=stage.COLOR_REM, alpha=0.3)

            # NREM
            y    = stagetime_circadian_profile_stats_list[g_idx][0, 1, :]
            y_sem= stagetime_circadian_profile_stats_list[g_idx][1, 1, :]/np.sqrt(num)
            ax2.plot(x, y, color=stage.COLOR_NREM)
            ax2.fill_between(x, y - y_sem,
                                y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            # Wake
            y    = stagetime_circadian_profile_stats_list[g_idx][0, 2, :]
            y_sem = stagetime_circadian_profile_stats_list[g_idx][1, 2, :]/np.sqrt(num)
            ax3.plot(x, y, color=stage.COLOR_WAKE)
            ax3.fill_between(x, y - y_sem,
                                y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

            fig.suptitle(f'{mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'stage-time_circadian_profile_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}.jpg'
            fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--faster2_dirs", required=True, nargs="*",
                        help="paths to the FASTER2 directoryies")
    parser.add_argument("-o", "--output_dir",
                        help="a path to the output files")

    args = parser.parse_args()

    faster_dir_list = args.faster2_dirs
    output_dir = args.output_dir
    if output_dir == None:
        # output to the current directory if not specified
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # collect mouse_infos of the specified (multiple) FASTER dirs
    mouse_info_collected = collect_mouse_info_df(faster_dir_list)
    mouse_info_df = mouse_info_collected['mouse_info']
    epoch_num = mouse_info_collected['epoch_num']

    # prepare stagetime statistics
    stagetime_stats = make_summary_stats(mouse_info_df, epoch_num)

    # draw stagetime profile of individual mice
    draw_stagetime_profile_individual(stagetime_stats, output_dir)
 
    # draw stagetime profile of grouped mice
    draw_stagetime_profile_grouped(stagetime_stats, output_dir)
    
    # draw stagetime circadian profile of individual mice
    draw_stagetime_circadian_profile_indiviudal(stagetime_stats, output_dir)

    # draw stagetime circadian profile of groups
    draw_stagetime_circadian_profile_grouped(stagetime_stats, output_dir)
