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

import faster2lib.eeg_tools as et
import stage

DOMAIN_NAMES = ['Slow', 'Delta w/o slow', 'Delta', 'Theta']


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
        device_label = r['Device label'].strip()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        stats_report = r['Stats report'].strip().upper()
        note = r['Note']
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']
        if stats_report=='NO':
            print(f'[{i+1}] skipping stage: {faster_dir} {device_label}')
            continue
        
        # read a stage file
        print(f'[{i+1}] reading stage: {faster_dir} {device_label} {stage_ext}')
        stage_call = et.read_stages(os.path.join(faster_dir, 'result'), device_label, stage_ext)
        stage_call = stage_call[epoch_range]
        epoch_num = len(stage_call)
        
        # stagetime in a day
        rem, nrem, wake, unknown = stagetime_in_a_day(stage_call)
        stagetime_df = stagetime_df.append([[exp_label, mouse_group, mouse_id, device_label, rem, nrem, wake, unknown, stats_report, note]], ignore_index=True)
        
        # stage time profile
        stagetime_profile_list.append(stagetime_profile(stage_call))
        
        # stage circadian profile
        stagetime_circadian_profile_list.append(stagetime_circadian_profile(stage_call))
        
        # transition matrix
        transmat_list.append(transmat_from_stages(stage_call))

        # sw transition
        swtrans_list.append(swtrans_from_stages(stage_call))
        
    stagetime_df.columns = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label', 'REM', 'NREM', 'Wake', 'Unknown', 'Stats report', 'Note']
    
    return({'stagetime': stagetime_df,
            'stagetime_profile': stagetime_profile_list,
            'stagetime_circadian': stagetime_circadian_profile_list,
            'transmat': transmat_list,
            'swtrans': swtrans_list,
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


def test_two_sample(x, y):
    ## test.two.sample: Performs two-sample statistical tests according to our labratory's standard.
    ##                          
    ## Arguments:
    ##  x: first samples
    ##  y: second samples
    ## 
    ## Return:
    ##  A dict of (p.value=p.value, method=method (string))
    ##
    
    # remove nan
    xx = np.array(x)
    yy = np.array(y)
    xx = xx[~np.isnan(xx)]
    yy = yy[~np.isnan(yy)]
    
    # If input data length < 2, any test is not applicable.
    if (len(xx)<2) or (len(yy)<2):
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
        
        equal_variance_p = var_test(xx,yy)['p_value']
            
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
        if not np.isnan(p_value) and p_value<0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = ''
    
    res = {'p_value': p_value, 'stars':stars, 'method':method}
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
    if F>1:
        p_value = stats.f.sf(F, df2, df1)*2 # two-sided
    else:
        p_value = (1-stats.f.sf(F, df2, df1))*2 # two-sided
     
    return {'F':F, 'df1':df1, 'df2':df2, 'p_value':p_value}


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
    epoch_num = stagetime_stats['epoch_num']
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
    epoch_num = stagetime_stats['epoch_num']
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


def x_shifts(values, y_min, y_max, width):
#    print(y_min, y_max)
    counts, _ = np.histogram(values,range=(np.min([y_min, np.min(values)]), np.max([y_max, np.max(values)])), bins=30)
    sorted_values = sorted(values)
    shifts = []
#    print(counts)
    non_zero_counts = counts[counts>0]
    for c in non_zero_counts:
        if c==1:
            shifts.append(0)
        else:
            p = np.arange(1,c+1) # point counts
            s = np.repeat(p, 2)[:p.size] * (-1)**p * width/10 # [-1, 1, -2, 2, ...] * width/10
            shifts.extend(s)
    
#     print(shifts)
#     print(sorted_values)
    return [np.array(shifts), sorted_values]


def scatter_datapoints(ax, w, x_pos, values):
    s, v =  x_shifts(values, *ax.get_ylim(), w)
    ax.scatter(x_pos + s, v , color='darkgrey')


def draw_stagetime_barchart(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']

    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups==g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(10,4))
    fig.subplots_adjust(wspace=0.5)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    w = 0.8 # bar width
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

    if num_groups>1:
        # REM
        values_c = stagetime_df['REM'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c  = np.std(values_c)/np.sqrt(len(values_c))
        ax1.bar(x_pos[0], mean_c, yerr=sem_c, align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax1, w, x_pos[0], values_c)
        for g_idx in range(1, num_groups):
            values_t = stagetime_df['REM'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t  = np.std(values_t)/np.sqrt(len(values_t))
            ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
            scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        #NREM
        values_c = stagetime_df['NREM'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c  = np.std(values_c)/np.sqrt(len(values_c))
        ax2.bar(x_pos[0], mean_c, yerr=sem_c, align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax2, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = stagetime_df['NREM'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t  = np.std(values_t)/np.sqrt(len(values_t))
            ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
            scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

        #Wake
        values_c = stagetime_df['Wake'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c  = np.std(values_c)/np.sqrt(len(values_c))
        ax3.bar(x_pos[0], mean_c, yerr=sem_c, align='center', width=w, capsize=6, color='grey', alpha=0.6)
        scatter_datapoints(ax3, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = stagetime_df['Wake'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t  = np.std(values_t)/np.sqrt(len(values_t))
            ax3.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
            scatter_datapoints(ax3, w, x_pos[g_idx], values_t)
    else:
        # REM
        values_t = stagetime_df['REM'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t  = np.std(values_t)/np.sqrt(len(values_t))
        ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center', width=w, capsize=6, color=stage.COLOR_REM, alpha=0.6)
        scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        #NREM
        values_t = stagetime_df['NREM'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t  = np.std(values_t)/np.sqrt(len(values_t))
        ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

        #Wake
        values_t = stagetime_df['Wake'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t  = np.std(values_t)/np.sqrt(len(values_t))
        ax3.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax3, w, x_pos[g_idx], values_t)

    fig.suptitle('Stage-times')
    filename = 'stage-time_barchart.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


def _draw_transition_barchart(mouse_groups, transmat_mat):
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups==g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(12,8))
    fig.subplots_adjust(wspace=0.2)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    w = 0.8 # bar width
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
    rr_vals_c = transmat_mat[bidx_group_list[0]][:,0,0]
    nn_vals_c = transmat_mat[bidx_group_list[0]][:,1,1]
    ww_vals_c = transmat_mat[bidx_group_list[0]][:,2,2]
    rw_vals_c = transmat_mat[bidx_group_list[0]][:,0,1]
    rn_vals_c = transmat_mat[bidx_group_list[0]][:,0,2]
    wr_vals_c = transmat_mat[bidx_group_list[0]][:,1,0]
    wn_vals_c = transmat_mat[bidx_group_list[0]][:,1,2]
    nr_vals_c = transmat_mat[bidx_group_list[0]][:,2,0]
    nw_vals_c = transmat_mat[bidx_group_list[0]][:,2,1]

    if num_groups > 1:
        # staying
        ## control
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
        ## control
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
        ## control
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
        ## control
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
        ## single group
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
        ## single group
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
        ## single group
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
        ## single group
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
    fig.suptitle(f'transition probability: {mouse_groups_set[0]} v.s. {"  ".join(mouse_groups_set[1:])}')
    filename = 'transition probability_barchart.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


def _odd(p, epoch_num):
    min_p = 1/epoch_num # zero probability is replaced by this value
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
    fig.suptitle(f'transition probability (log odds): {mouse_groups_set[0]} v.s. {"  ".join(mouse_groups_set[1:])}')
    filename = 'transition probability_barchart_logodds.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


def _draw_swtransition_barchart(mouse_groups, swtrans_mat):
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups==g for g in mouse_groups_set]
    num_groups = len(mouse_groups_set)

    fig = Figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    w = 0.8 # bar width
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(['Psw', 'Pws'])

    # control group (always index: 0)
    num_c = np.sum(bidx_group_list[0])
    sw_vals_c = swtrans_mat[bidx_group_list[0]][:,0]
    ws_vals_c = swtrans_mat[bidx_group_list[0]][:,1]

    ## Psw and Pws
    x_pos = 0 - w/2
    ax.bar(x_pos, 
            height=np.mean(sw_vals_c), 
            yerr  =np.std(sw_vals_c)/num_c, 
            align='center', width=w, capsize=6, color='gray', alpha=0.6)
    scatter_datapoints(ax, w, x_pos, sw_vals_c)
    x_pos = 2 - w/2
    ax.bar(x_pos, 
            height=np.mean(ws_vals_c), 
            yerr  =np.std(ws_vals_c)/num_c, 
            align='center', width=w, capsize=6, color='gray', alpha=0.6)
    scatter_datapoints(ax, w, x_pos, ws_vals_c)

    # test group index: g_idx.
    for g_idx in range(1, num_groups):
        num_t = np.sum(bidx_group_list[g_idx])
        sw_vals_t = swtrans_mat[bidx_group_list[g_idx]][:,0]
        x_pos = 0 + g_idx*w/2
        ax.bar(x_pos, 
                height=np.mean(sw_vals_t), 
                yerr  =np.std(sw_vals_t)/num_t, 
                align='center', width=w, capsize=6, color=stage.COLOR_NREM, alpha=0.6)
        scatter_datapoints(ax, w, x_pos, sw_vals_t)

        ws_vals_t = swtrans_mat[bidx_group_list[g_idx]][:,1]
        x_pos = 2 + g_idx*w/2
        ax.bar(x_pos, 
                height=np.mean(ws_vals_t), 
                yerr  =np.std(ws_vals_t)/num_t, 
                align='center', width=w, capsize=6, color=stage.COLOR_WAKE, alpha=0.6)
        scatter_datapoints(ax, w, x_pos, ws_vals_t)

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
    fig.suptitle(f'sleep/wake trantision probability:\n {mouse_groups_set[0]} v.s. {"  ".join(mouse_groups_set[1:])}')
    filename = 'sleep-wake transition probability_barchart.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


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
    fig.suptitle(f'sleep/wake trantision probability (log odds) :\n {mouse_groups_set[0]} v.s. {"  ".join(mouse_groups_set[1:])}')
    filename = 'sleep-wake transition probability_barchart_logodds.jpg'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)


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

def make_log_psd_profile(mouse_info_df, sample_freq, epoch_range, stage_ext):
    """makes summary log PSD statics of each mouse:
            psd_mean_df: mean log (decibel like) PSD profiles of each stage for each mice.
            This function is same with make_psd_profile() except the log transformation. 
   
    Arguments:
        mouse_info_df {pd.DataFram} -- an dataframe given by mouse_info_collected()
        sample_freq {int} -- sampling frequency
        epoch_range {slice} -- a range of target epochs
        stage_ext {str} -- a file sub-extention (e.g. 'faster2' for *.faster2.csv)

    Returns:
        psd_mean_df {pd.DataFrame} --  Experiment label, Mouse group, Mouse ID, 
            Device label, Stage, [freq_bins...]

    """

    psd_mean_df = pd.DataFrame()
    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label'].strip()
        stats_report = r['Stats report'].strip().upper()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']


        if stats_report == 'NO':
            print(f'[{i+1}] skipping: {faster_dir} {device_label}')
            continue
        print(f'[{i+1}] reading PSD and stage of: {faster_dir} {device_label}')

        # read stage of the mouse
        stage_call = et.read_stages(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)

        # read the normalized EEG PSDs and the associated normalization factors and means
        pkl_path_eeg = os.path.join(
            faster_dir, 'result', 'PSD', f'{device_label}_EEG_PSD.pkl')
        with open(pkl_path_eeg, 'rb') as pkl:
            snorm_psd = pickle.load(pkl)

        # convert the spectrum normalized PSD to the log PSD
        conv_psd = 10*np.log10(conv_PSD_from_snorm_PSD(snorm_psd)) # this is the only difference to make_psd_profile()

        # Break at the error: the unknown PSD is not recoverable
        bidx_unknown_psd = snorm_psd['bidx_unknown']
        if not np.all(stage_call[bidx_unknown_psd] == 'UNKNOWN'):
            print('[Error] "unknown" is recoverable')
            idx = list(np.where(bidx_unknown_psd)[0][stage_call[bidx_unknown_psd] != 'UNKNOWN'])
            print(f'... in stage file at {idx}')
            break
        
        # remove unknown-PSD epoch from stage_call
        stage_call = stage_call[~bidx_unknown_psd]

        # bidx_target: bidx for the selected range 
        bidx_selected = np.repeat(False, len(bidx_unknown_psd))
        bidx_selected[epoch_range] = True 
        bidx_target = bidx_selected[~bidx_unknown_psd]

        bidx_rem =  (stage_call == 'REM') & bidx_target
        bidx_nrem = (stage_call == 'NREM') & bidx_target
        bidx_wake = (stage_call == 'WAKE') & bidx_target
        psd_mean_rem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_rem, :])
        psd_mean_nrem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_nrem, :])
        psd_mean_wake = np.apply_along_axis(np.mean, 0, conv_psd[bidx_wake, :])

        psd_mean_df = psd_mean_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'REM'] + psd_mean_rem.tolist()], ignore_index=True)
        psd_mean_df = psd_mean_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'NREM'] + psd_mean_nrem.tolist()], ignore_index=True)
        psd_mean_df = psd_mean_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'Wake'] + psd_mean_wake.tolist()], ignore_index=True)

    # frequency bins
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    freq_columns = [f'f@{x}' for x in freq_bins.tolist()]
    column_names = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label', 'Stage'] + freq_columns
    psd_mean_df.columns = column_names
    return psd_mean_df


def make_psd_profile(mouse_info_df, sample_freq, epoch_range, stage_ext):
    """makes summary PSD statics of each mouse:
            psd_mean_df: mean PSD profiles of each stage for each mice
   
    Arguments:
        mouse_info_df {pd.DataFram} -- an dataframe given by mouse_info_collected()
        sample_freq {int} -- sampling frequency
        epoch_range {slice} -- a range of target epochs
        stage_ext {str} -- a file sub-extention (e.g. 'faster2' for *.faster2.csv)

    Returns:
        psd_mean_df {pd.DataFrame} --  Experiment label, Mouse group, Mouse ID, 
            Device label, Stage, [freq_bins...]

    """

    psd_mean_df = pd.DataFrame()
    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label'].strip()
        stats_report = r['Stats report'].strip().upper()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']


        if stats_report == 'NO':
            print(f'[{i+1}] skipping: {faster_dir} {device_label}')
            continue
        print(f'[{i+1}] reading PSD and stage of: {faster_dir} {device_label}')

        # read stage of the mouse
        stage_call = et.read_stages(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)

        # read the normalized EEG PSDs and the associated normalization factors and means
        pkl_path_eeg = os.path.join(
            faster_dir, 'result', 'PSD', f'{device_label}_EEG_PSD.pkl')
        with open(pkl_path_eeg, 'rb') as pkl:
            snorm_psd = pickle.load(pkl)

         # convert the spectrum normalized PSD to the conventional PSD
        conv_psd = conv_PSD_from_snorm_PSD(snorm_psd)

        # Break at the error: the unknown PSD is not recoverable
        bidx_unknown_psd = snorm_psd['bidx_unknown']
        if not np.all(stage_call[bidx_unknown_psd] == 'UNKNOWN'):
            print('[Error] "unknown" is recoverable')
            idx = list(np.where(bidx_unknown_psd)[0][stage_call[bidx_unknown_psd] != 'UNKNOWN'])
            print(f'... in stage file at {idx}')
            break

        # remove unknown-PSD epoch from stage_call
        stage_call = stage_call[~bidx_unknown_psd]

        # bidx_target: bidx for the selected range 
        bidx_selected = np.repeat(False, len(bidx_unknown_psd))
        bidx_selected[epoch_range] = True 
        bidx_target = bidx_selected[~bidx_unknown_psd]
    
        bidx_rem =  (stage_call == 'REM') & bidx_target
        bidx_nrem = (stage_call == 'NREM') & bidx_target
        bidx_wake = (stage_call == 'WAKE') & bidx_target
        psd_mean_rem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_rem, :])
        psd_mean_nrem = np.apply_along_axis(np.mean, 0, conv_psd[bidx_nrem, :])
        psd_mean_wake = np.apply_along_axis(np.mean, 0, conv_psd[bidx_wake, :])

        psd_mean_df = psd_mean_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'REM'] + psd_mean_rem.tolist()], ignore_index=True)
        psd_mean_df = psd_mean_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'NREM'] + psd_mean_nrem.tolist()], ignore_index=True)
        psd_mean_df = psd_mean_df.append([
            [exp_label, mouse_group, mouse_id, device_label, 'Wake'] + psd_mean_wake.tolist()], ignore_index=True)

    # frequency bins
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    freq_columns = [f'f@{x}' for x in freq_bins.tolist()]
    column_names = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label', 'Stage'] + freq_columns
    psd_mean_df.columns = column_names
    return psd_mean_df


def draw_PSDs_individual(psd_profiles_df, sample_freq, output_dir, opt_label=''):
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    # mouse_set
    mouse_list = psd_profiles_df['Mouse ID'].tolist()
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)  # unique elements with preseved order

    # draw individual PSDs
    for m in mouse_set:
        fig = Figure(figsize=(16, 4))
        fig.subplots_adjust(wspace=0.25)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel('REM\nnormalized PSD [AU]')
        ax2.set_xlabel('freq. [Hz]')
        ax2.set_ylabel('NREM\nnormalized PSD [AU]')
        ax3.set_xlabel('freq. [Hz]')
        ax3.set_ylabel('Wake\nnormalized PSD [AU]')

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
        filename = f'{opt_label}PSD_{"_".join(mouse_tag_list)}.jpg'
        fig.savefig(os.path.join(output_dir, filename), pad_inches=0,
                    bbox_inches='tight', dpi=100, quality=85, optimize=True)


def draw_PSDs_group(psd_profiles_df, sample_freq, output_dir, opt_label=''):
    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    # mouse_group_set
    mouse_group_list = psd_profiles_df['Mouse group'].tolist()
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)  # unique elements with preseved order

    # draw gropued PSD
    ## _c of Control (assuming index = 0 is a control mouse)
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

    for g_idx in range(1, len(mouse_group_set)):
        fig = Figure(figsize=(16, 4))
        fig.subplots_adjust(wspace=0.25)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel('REM\nnormalized PSD [AU]')
        ax2.set_xlabel('freq. [Hz]')
        ax2.set_ylabel('NREM\nnormalized PSD [AU]')
        ax3.set_xlabel('freq. [Hz]')
        ax3.set_ylabel('Wake\nnormalized PSD [AU]')

        ## _t of Treatment
        df = psd_profiles_df[psd_profiles_df['Mouse group'] == mouse_group_set[g_idx]]
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

        x = freq_bins
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
        filename = f'{opt_label}PSD_{mouse_group_set[0]}_vs_{mouse_group_set[g_idx]}.jpg'
        fig.savefig(os.path.join(output_dir, filename), pad_inches=0,
                    bbox_inches='tight', dpi=100, quality=85, optimize=True)


def write_sleep_stats(stagetime_stats, output_dir):
    stagetime_df = stagetime_stats['stagetime']
    mouse_groups = stagetime_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]

    sleep_stats_df = pd.DataFrame() # mouse_group, stage_type, num, mean, SD, pvalue, star, method

    # mouse_group's index:0 is always control
    mg = mouse_groups_set[0]
    bidx = bidx_group_list[0]
    num = np.sum(bidx)
    rem_values_c  = stagetime_df['REM'].values[bidx]
    nrem_values_c = stagetime_df['NREM'].values[bidx]
    wake_values_c = stagetime_df['Wake'].values[bidx]
    row1 = [mg, 'REM',  num, np.mean(rem_values_c),  np.std(rem_values_c),  np.nan, None, None]
    row2 = [mg, 'NREM', num, np.mean(nrem_values_c), np.std(nrem_values_c), np.nan, None, None]
    row3 = [mg, 'Wake', num, np.mean(wake_values_c), np.std(wake_values_c), np.nan, None, None]

    sleep_stats_df = sleep_stats_df.append([row1, row2, row3])
    for i, bidx in enumerate(bidx_group_list[1:]):
        idx = i+1
        mg = mouse_groups_set[idx]
        bidx = bidx_group_list[idx]
        num = np.sum(bidx)
        rem_values_t  = stagetime_df['REM'].values[bidx]
        nrem_values_t = stagetime_df['NREM'].values[bidx]
        wake_values_t = stagetime_df['Wake'].values[bidx]
        
        tr = test_two_sample(rem_values_c,  rem_values_t) # test for REM
        tn = test_two_sample(nrem_values_c, nrem_values_t) # test for NREM
        tw = test_two_sample(wake_values_c, wake_values_t) # test for Wake
        row1 = [mg, 'REM',  num, np.mean(rem_values_t),  np.std(rem_values_t),  tr['p_value'], tr['stars'], tr['method']]
        row2 = [mg, 'NREM', num, np.mean(nrem_values_t), np.std(nrem_values_t), tn['p_value'], tn['stars'], tn['method']]
        row3 = [mg, 'Wake', num, np.mean(wake_values_t), np.std(wake_values_t), tw['p_value'], tw['stars'], tw['method']]

        sleep_stats_df = sleep_stats_df.append([row1, row2, row3])
        
    sleep_stats_df.columns = ['Mouse group', 'Stage type', 'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    sleep_stats_df.to_csv(os.path.join(output_dir, 'stage-time_stats_table.csv'), index=False)
    stagetime_df.to_csv(os.path.join(output_dir, 'stage-time_table.csv'), index=False)


def make_psd_domain(psd_profiles_df):
    """ makes PSD power averaged within frequency domains of each stage for each mice

    Arguments:
        psd_profiles_df {pd.DataFrame} -- a dataframe given by make_psd_profile()

    Returns:
        [pd.DataFrame] -- Experiment label, Mouse group, Mouse ID, 
            Device label, Stage, Slow, Delta w/o slow, Delta, Theta
    """
    # get freq_bins from column names
    freq_bin_columns = psd_profiles_df.columns[5:].tolist()
    freq_bins = np.array([float(x.strip().split('@')[1]) for x in freq_bin_columns])
    
    # frequency domains
    bidx_theta_freq = (freq_bins>=4) & (freq_bins<10) # 15 bins
    bidx_delta_freq = (freq_bins<4) # 11 bins
    bidx_delta_wo_slow_freq =  (1<=freq_bins) & (freq_bins<4) # 8 bins (delta without slow)
    bidx_slow_freq = (freq_bins<1) # 3 bins

    # make psd_domain_df
    row_list = []
    for _, r in psd_profiles_df.iterrows():
        infos = r[:5]
        powers = r[5:]
        powers_slow = powers[bidx_slow_freq]
        powers_delta_wo_slow = powers[bidx_delta_wo_slow_freq]
        powers_delta = powers[bidx_delta_freq]
        powers_theta = powers[bidx_theta_freq]
        
        slow_p = np.mean(powers_slow)
        delta_wo_slow_p = np.mean(powers_delta_wo_slow)
        delta_p = np.mean(powers_delta)
        theta_p = np.mean(powers_theta)
        domain_powers = pd.Series([slow_p, delta_wo_slow_p, delta_p, theta_p], index=DOMAIN_NAMES)

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
        bidx_rem  =  (psd_domain_df['Stage'] == 'REM')
        bidx_nrem =  (psd_domain_df['Stage'] == 'NREM')
        bidx_wake =  (psd_domain_df['Stage'] == 'Wake')

        domain_powers_rem  = psd_domain_df.loc[bidx_group & bidx_rem ][DOMAIN_NAMES]
        domain_powers_nrem = psd_domain_df.loc[bidx_group & bidx_nrem][DOMAIN_NAMES]
        domain_powers_wake = psd_domain_df.loc[bidx_group & bidx_wake][DOMAIN_NAMES]

        return [domain_powers_rem, domain_powers_nrem, domain_powers_wake]
    

    psd_stats_df = pd.DataFrame() 
    # mouse_group_set
    mouse_group_list = psd_domain_df['Mouse group'].tolist()
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)  # unique elements with preseved order

    # control
    group_c = mouse_group_set[0] # index=0 should be always control group

    ## There are 3 powers_domains_[stages] where [stages] are [REM, NREM, Wake].
    ## Each powers_domains_[stage] contains 4 Series of domain powers: 
    ## [slow x mice, delta wo slow x mice, delta x mice, theta x mice]
    ## Therefore, the following loop results in 12 rows.
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

    ## treatment
    for group_t in mouse_group_set[1:]:
        rows = []
        powers_domains_stages_t = _domain_powers_by_group(psd_domain_df, group_t)
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


def write_psd_stats(psd_profiles_df, output_dir, opt_label=''):
    """ writes three PSD tables:
        1. psd_profile.csv: mean PSD profile of each stage for each mice
        2. psd_freq_domain_table.csv: PSD power averaged within frequency domains of each stage for each mice 
        3. psd_stats_table.csv: statistical tests for each frequency domains between groups 
    """

    psd_domain_df = make_psd_domain(psd_profiles_df)
    psd_stats_df = make_psd_stats(psd_domain_df)

    # write tabels
    psd_profiles_df.to_csv(os.path.join(output_dir, f'{opt_label}PSD_profile.csv'), index=False)
    psd_domain_df.to_csv(os.path.join(output_dir, f'{opt_label}PSD_freq_domain_table.csv'), index=False)
    psd_stats_df.to_csv(os.path.join(output_dir, f'{opt_label}PSD_stats_table.csv'), index=False)


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
        e_range = [int(x.strip()) if x else None for x in args.epoch_range.split(':')]
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
        if len(faster_dir_list)>1:
            basenames = [os.path.basename(dir_path) for dir_path in faster_dir_list]
            path_ext = '_' + '_'.join(basenames)
        else:
            path_ext = ''    
        output_dir = os.path.join(faster_dir_list[0], 'summary' + path_ext)
    os.makedirs(output_dir, exist_ok=True)

    # prepare stagetime statistics
    stagetime_stats = make_summary_stats(mouse_info_df, epoch_range, stage_ext)

    # write a table of stats
    write_sleep_stats(stagetime_stats, output_dir)

    # # draw stagetime profile of individual mice
    draw_stagetime_profile_individual(stagetime_stats, output_dir)
 
    # draw stagetime profile of grouped mice
    draw_stagetime_profile_grouped(stagetime_stats, output_dir)
    
    # draw stagetime circadian profile of individual mice
    draw_stagetime_circadian_profile_indiviudal(stagetime_stats, output_dir)

    # draw stagetime circadian profile of groups
    draw_stagetime_circadian_profile_grouped(stagetime_stats, output_dir)

    # draw stagetime barchart
    draw_stagetime_barchart(stagetime_stats, output_dir)

    # draw transition barchart (probability)
    draw_transition_barchart_prob(stagetime_stats, output_dir)

    # draw transition barchart (log odds)
    draw_transition_barchart_logodds(stagetime_stats, output_dir)

    # draw sleep/wake transition probability
    draw_swtransition_barchart_prob(stagetime_stats, output_dir)

    # draw sleep/wake transition probability (log odds)
    draw_swtransition_barchart_logodds(stagetime_stats, output_dir)

    # prepare Powerspectrum density (PSD) profiles for individual mice
    psd_profiles_df = make_psd_profile(mouse_info_df, sample_freq, epoch_range, stage_ext)
    log_psd_profiles_df = make_log_psd_profile(mouse_info_df, sample_freq, epoch_range, stage_ext) # decebel-like

    # write a table of PSD
    write_psd_stats(psd_profiles_df, output_dir)
    write_psd_stats(log_psd_profiles_df, output_dir, 'log-')

    # draw power density plot
    draw_PSDs_individual(psd_profiles_df, sample_freq, output_dir)
    draw_PSDs_individual(log_psd_profiles_df, sample_freq, output_dir, 'log-')
    draw_PSDs_group(psd_profiles_df, sample_freq, output_dir)
    draw_PSDs_group(log_psd_profiles_df, sample_freq, output_dir, 'log-')