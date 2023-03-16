# -*- coding: utf-8 -*-
""" Performs the additional analysis on the delta-power dynamics(dpd)
    based on results of summary.py
"""
import os
import sys
import argparse
import json
import copy
from datetime import datetime
import itertools
import logging
import textwrap
from logging import getLogger, StreamHandler, FileHandler, Formatter
import pandas as pd
import numpy as np
from matplotlib import patches
from matplotlib.figure import Figure
from scipy import optimize
from scipy import stats
from scipy import linalg
from sklearn.metrics import r2_score
import stage
import faster2lib.eeg_tools as et
import faster2lib.summary_common as sc
import summary

EPISODE_LEN_MINUTES = 5
COLOR_SERIES = ['grey', '#D81B60', '#FFC107', '#1E88E5',
                '#004D40']  # grey, red, orange, blue, green


def initialize_logger(log_file):
    """ initializes the logger
    """
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
    """ print message to log if defined
    """
    if 'log' in globals():
        log.info(msg)
    else:
        print(msg)


def print_log_exception(msg):
    """ print message to log.exception if defined
    """

    if 'log' in globals():
        log.exception(msg)
    else:
        print(msg)


def read_delta_power_csv(path_to_delta_power_csv):
    """ read a CSV file of delta powers given by summary.py

        Arguments:
            path_to_delta_power_csv: A string path to the CSV file

        Returns:
            Two dataframes from the CSV file: header and body. These dataframes are
            supposed to be processed by extract_delta_power()
    """
    print_log(
        f'Reading the delta power from the csv file: {path_to_delta_power_csv}')
    csv_head = pd.read_csv(path_to_delta_power_csv, nrows=4, header=None)
    csv_body = pd.read_csv(path_to_delta_power_csv, skiprows=4, header=None)
    return (csv_head, csv_body)


def select_delta_power(csv_head, csv_body, keys: dict):
    """ extract a column of delta-power in csv_body by a keys in csv_header

        Arguments:
            csv_head: a dataframe given by read_delta_power_csv()
            csv_body: a dataframe givem by read_delta_power_csv()
            keys: a dict that specifies a column of delta-power

        Returns:
            numpy array of delta powers

        Raise: LookupError when keys failed to specify only one column

    """
    all_true = np.repeat(True, csv_head.shape[1])
    bidx_exp_label = bidx_mouse_group = bidx_mouse_id = bidx_device_label = all_true

    if 'Experiment label' in keys.keys():
        bidx_exp_label = (csv_head.iloc[0] == keys['Experiment label'])
    if 'Mouse group' in keys.keys():
        bidx_mouse_group = (csv_head.iloc[1] == keys['Mouse group'])
    if 'Mouse ID' in keys.keys():
        bidx_mouse_id = (csv_head.iloc[2] == keys['Mouse ID'])
    if 'Device label' in keys.keys():
        bidx_device_label = (csv_head.iloc[3] == keys['Device label'])
    bidx_key_match = bidx_exp_label & bidx_mouse_group & bidx_mouse_id & bidx_device_label

    if np.sum(bidx_key_match) == 0:
        raise LookupError(f'No match found for {keys} in the delta power csv file. '
                          'Check the CSV header contains a column with the keys.')
    if np.sum(bidx_key_match) > 1:
        raise LookupError(f'Multiple match found for {keys} in the delta power csv file. '
                          'Check the CSV header contains only one column with the keys.')

    idx_col = np.where(bidx_key_match)[0][0]
    delta_power = csv_body[idx_col].to_numpy()
    return delta_power


def flatten(list_2d):
    """flattens the given 2D list to 1D list
    """
    return list(itertools.chain.from_iterable(list_2d))


def breakup_size(size, episode_len_epoch):
    num_segment = np.floor(size/episode_len_epoch)
    if num_segment < size/episode_len_epoch:
        broken_sizes = np.concatenate([
            np.repeat(episode_len_epoch, size/episode_len_epoch),
            [int(size - episode_len_epoch*num_segment)]
        ])
    else:
        broken_sizes = np.repeat(episode_len_epoch, size/episode_len_epoch)
    return broken_sizes.tolist()


def degrade_stage(stage_call):
    ''' make two-level stage calls (D, I) from the three-level
    stage calls(WAKE, REM, NREM)

    '''
    # Note: stage 'I' includes Unknown etc. because unknown stages are likely from wakefulness
    two_stage_call = np.array(
        ['D' if stage == 'NREM' else 'I' for stage in stage_call])

    return two_stage_call


def make_episode(stage_call, episode_len):
    """meakes an array of episodes

    Args:
        stage_call (np.array): stage call of NREM, WAKE, and REM
        episode_len (int): episode length in epoch

    Returns:
        tuple of arrays: the stages, the size, and the starting epoch index of each episode
    """
    # Two level stages: NREM=>D, REM & WAKE=>I
    two_stage_call = degrade_stage(stage_call)

    # Find indices (idx) of epohcs where the stage changes
    bidx_two_stage_change = two_stage_call[1:] != two_stage_call[:-1]
    # The first item is always a 'change'
    bidx_two_stage_change = np.concatenate([[True], bidx_two_stage_change])
    idx_stage_change = np.where(bidx_two_stage_change)[0]

    # sblock is a continuous epochs of sustained stages
    sblock_size = idx_stage_change[1:] - idx_stage_change[:-1]
    sblock_size = np.concatenate(
        [sblock_size, [len(stage_call)-idx_stage_change[-1]]])
    sblock_stage = two_stage_call[bidx_two_stage_change]

    # break up the D sblock into D episodes if it is longer than EPISODE_LEN_MINUITES
    episode_size_list = [breakup_size(size, episode_len) if (size > episode_len and stage == 'D') else [size]
                         for size, stage in zip(sblock_size, sblock_stage)]
    episode_size = np.array(flatten(episode_size_list))

    episode_stage_list = [np.repeat(stage, len(sizes)).tolist()
                          for stage, sizes in zip(sblock_stage, episode_size_list)]
    episode_stage = np.array(flatten(episode_stage_list))

    idx_episode_start_list = [(idx_start + np.arange(0, size, episode_len)).tolist() if (size > episode_len and stage == 'D') else [idx_start]
                              for idx_start, size, stage in zip(idx_stage_change, sblock_size, sblock_stage)]
    idx_episode_start = np.array(flatten(idx_episode_start_list))

    return (episode_stage, episode_size, idx_episode_start)


def make_D_episode(delta_power, episode_size, episode_len, episode_stage, idx_episode_start):
    # Since we devide longer D-episodes into episode with the maximum length of EPISODE_LEN_MINUTES, the residual episode
    # shorter than EPISODE_LEN_MINUTES is excluded in the subsequent steps.
    bidx_D_episode = (episode_size >= episode_len) & (episode_stage == 'D')
    idx_D_episode = idx_episode_start[bidx_D_episode]
    size_D_episode = episode_size[bidx_D_episode]
    num_of_D_episode = len(idx_D_episode)

    delta_power_D_episode = np.array([empty_median(delta_power[start_idx:(start_idx+episode_len)])
                                      for start_idx, episode_len in zip(idx_D_episode, size_D_episode)])

    return (num_of_D_episode, delta_power_D_episode, idx_D_episode, bidx_D_episode)


def _draw_asymptotes_histogram(delta_power_R, delta_power_W, delta_power_N, kernel_R, kernel_W, kernel_N, low_asymp, up_asymp):
    """ draws the histogram of epoch delta-power in each stage

    Args:
        delta_power_R (np.array): delta powers of epochs in REM sleep
        delta_power_W (np.array): delta powers of epochs in Wakefulness
        delta_power_N (np.array): delta powers of epochs in NREM sleep
        kernel_R (gaussian_kde): An object given by stats.gaussian_kde in estimate_delta_power_asympotete()
        kernel_W (gaussian_kde): An object given by stats.gaussian_kde in estimate_delta_power_asympotete()
        kernel_N (gaussian_kde): An object given by stats.gaussian_kde in estimate_delta_power_asympotete()
        low_asymp (float): 1-percent percentile of delta-powers in NREM sleep
        up_asymp (float): 99-percent percentile of delta-powers in NREM sleep

    Returns:
        Figure : A histogram
    """
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    x = np.linspace(np.nanmin(np.concatenate(
        [delta_power_W, delta_power_R])), np.nanmax(delta_power_N), 100)

    if kernel_R is not None:
        ax.plot(x, kernel_R(x), linewidth=3, color=stage.COLOR_REM, label='REM')
        _ = ax.hist(delta_power_R, bins=100, density=True,
                color=stage.COLOR_REM, alpha=0.7)

    ax.plot(x, kernel_W(x), linewidth=3, color=stage.COLOR_WAKE, label="Wake")
    _ = ax.hist(delta_power_W, bins=100, density=True,
                color=stage.COLOR_WAKE, alpha=0.7)

    ax.plot(x, kernel_N(x), linewidth=3, color=stage.COLOR_NREM, label="NREM")
    _ = ax.hist(delta_power_N, bins=100, density=True,
                color=stage.COLOR_NREM, alpha=0.7)

    ax.axvline(low_asymp, c='black')
    ax.axvline(up_asymp, c='black')

    ax.set_xlabel('Delta power [AU]')
    ax.set_ylabel('Density')
    ax.legend()

    return fig


def estimate_delta_power_asymptote(stage_call, delta_power):
    bidx_R = (stage_call == 'REM')
    bidx_W = (stage_call == 'WAKE')
    bidx_N = (stage_call == 'NREM')
    try:
        delta_power_R = delta_power[bidx_R]
        delta_power_W = delta_power[bidx_W]
        delta_power_N = delta_power[bidx_N]
    except IndexError as idx_err:
        print_log(f'[Error] There is a mismatch in the epoch lengths between stages ({len(stage_call)}) and delta powers ({len(delta_power)}).\n'
                   'Check the epoch length of the analysis or use -e option to specify it.\n')
        raise

    # If there are not multiple REM epochs, the kernel is None
    if len(delta_power_R)>1:
        kernel_R = stats.gaussian_kde(delta_power_R[~np.isnan(delta_power_R)])
    else:
        kernel_R = None

    kernel_W = stats.gaussian_kde(delta_power_W[~np.isnan(delta_power_W)])
    kernel_N = stats.gaussian_kde(delta_power_N[~np.isnan(delta_power_N)])

    low_asymp = np.nanpercentile(delta_power_N, 1)
    up_asymp = np.nanpercentile(delta_power_N, 99)

    fig = _draw_asymptotes_histogram(
        delta_power_R, delta_power_W, delta_power_N, kernel_R, kernel_W, kernel_N, low_asymp, up_asymp)

    return (low_asymp, up_asymp, fig)



def _set_common_features_delta_power_dynamics(ax, x_max, y_range):
    y_diff = (y_range[1] - y_range[0])
    # add 10% space over and 5% space under the y_range
    ymin = y_range[0] - 0.05 * y_diff
    ymax = y_range[1] + 0.1 * y_diff
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, x_max+1, 6))
    ax.set_yticks([round(x, 2) for x in np.linspace(max(0, ymin), ymax, 6)])
    ax.grid(dashes=(2, 2))
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Median delta power in NREM episode [%]')

    # add the light bar
    light_bar_base = patches.Rectangle(
        xy=[0, ymin], width=x_max, height=0.05*y_diff, fill=True, color=stage.COLOR_DARK)
    ax.add_patch(light_bar_base)
    for day in range(int(x_max/24)):
        light_bar_light = patches.Rectangle(
            xy=[24*day, ymin], width=12, height=0.05*y_diff, fill=True, color=stage.COLOR_LIGHT)
        ax.add_patch(light_bar_light)


def simulate_delta_power_dynamics(opt_taus, low_asymp, up_asymp, stage_call, idx_D_episode, delta_power_D_episode, delta_t):
    """ simulate the delta power dynamics with the given parameters. This function also returns the observed delta powers
    so that the returned observed delta powers can be simply plotted over the simulated delta powers.

    Args:
        opt_taus (tuple): Three parametes to be optimized; tau_i, tau_d, and s0
        low_asymp (float): The lower asymptote
        up_asymp (float): The upper asymptote
        stage_call (np.array): An array of stages (NREM, REM, WAKE)
        idx_D_episode (np.array): The array of starting-epoch indices of D-episode
        delta_power_D_episode (np.array): The array of delta-powers of D-episode
        delta_t (int): The time interval between stages (epoch_len_sec)

    Returns:
        tuble: Two arrays of the simulated and observed delta power time-series
    """

    # Two level stages: NREM=>D, REM & WAKE=>I
    two_stage_call = degrade_stage(stage_call)

    tau_i, tau_d, s0 = [opt_taus[0], opt_taus[1], opt_taus[2]]

    epoch_num = len(two_stage_call)
    simulated_s = np.zeros(epoch_num + 1)
    simulated_s[0] = s0
    bidx_D_epoch = (two_stage_call == 'D')

    for i, boolean_stage in zip(range(epoch_num), bidx_D_epoch):
        simulated_s[i+1] = (1-boolean_stage)*(up_asymp - (up_asymp - simulated_s[i])*np.exp(-delta_t/tau_i)) +\
            boolean_stage * \
            (low_asymp + (simulated_s[i] - low_asymp)*np.exp(-delta_t/tau_d))

    # return time-series of delta power dynamics both for simunation and observation
    delta_power_dynamics_simulated = copy.deepcopy(simulated_s[1:])
    delta_power_dynamics_observed = np.repeat(np.nan, epoch_num)
    delta_power_dynamics_observed[idx_D_episode] = delta_power_D_episode

    return (delta_power_dynamics_simulated, delta_power_dynamics_observed)


def draw_simulated_delta_power_dynamics(sim_ts, obs_ts, delta_t, y_range=None, epoch_range_basal=None):
    """draws similated timeseries of delta-power dynamics wiht observed data

    Args:
        sim_ts (np.array): The simulated delta-powers
        obs_ts (np.array): The observed delta-powers
        delta_t (int): The epoch length in second
        epoch_range_basal (slice): The slice used for selecting the basal epoch range

    Returns:
        Figure : A plot of the simulated time-series of delta-power dynamics
    """
    num_points = len(sim_ts)
    if y_range is None:
        y_min = np.nanmin([sim_ts, obs_ts])
        y_max = np.nanmax([sim_ts, obs_ts])
    else:
        y_min, y_max = y_range

    fig = Figure(figsize=(13, 6))
    ax = fig.add_subplot(111)

    if epoch_range_basal is not None:
        # Marked the extrapolated area with the pale red backgroud color
        ax.axvspan(0, num_points*delta_t/3600, color='r', alpha=0.1)
        ax.axvspan(epoch_range_basal.start*delta_t/3600, epoch_range_basal.stop*delta_t/3600, color='w')

    _set_common_features_delta_power_dynamics(ax, num_points*delta_t/3600, [y_min, y_max])

    bidx_valid_obs_ts = ~np.isnan(obs_ts)
    idx_valid_obs_ts = np.where(bidx_valid_obs_ts)[0]
    ax.plot(np.arange(num_points)*delta_t/3600, sim_ts, c=COLOR_SERIES[1])
    ax.scatter(idx_valid_obs_ts*delta_t/3600, obs_ts[bidx_valid_obs_ts], s=8)
    ax.set_xlim(0, num_points*delta_t/3600)

    return (fig)


def do_fitting(episode_stage, episode_size, bidx_D_episode, delta_power_D_episode, epoch_len_sec, low_asymp, up_asymp):
    """ Simulates the process with various pairs of (tau_i, tau_d) to find the best fit
    to the observed delta-power timeseries of D-stage episodes.

    Arguments:
        episode_stage {np.array} -- An array of stages of episode given by make_episode()
        episode_size {np.array} -- An array of sizes of episode given by make_episode()
        bidx_D_episode {np.array} -- An binary index for the episode array indicating D episodes
        delta_power_D_eipsode {np.array} -- An array of "observed" delta-powers of D-stage episode
        epoch_len_sec {int} -- Epoch length in seconds
        low_asymp {double} -- Lower asymptote of the S-process
        up_asymp {double} -- Upper asymptote of the S-process
    """
    # define private functions
    def _next_S_D(s, delta_t, tau_d):
        s_next = low_asymp + (s - low_asymp)*np.exp(-delta_t/tau_d)
        return s_next

    def _next_S_I(s, delta_t, tau_i):
        s_next = up_asymp - (up_asymp - s)*np.exp(-delta_t/tau_i)
        return s_next

    def _evaluate_process_s(params):
        """ Simulate process-S with the given parameters. Then evaluate the error between
        the simulation and the observed delta powers in D episodes.

            Arguments:
                params: [tau_i, tau_d, s0]

            Returns:
                The sum of squared differences wrapped by np.array()

            Variables from outer scope:
                The following variables should be defined in the outer scope

                episode_stage:
                episode_size:
                bidx_D_episode:
                delta_power_D_episode:
                scale:
                epoch_len_sec:

        """
        [tau_i, tau_d, s0] = params
        simulated_s[0] = s0

        # given as arguments of the outer function
        nonlocal episode_stage
        nonlocal episode_size
        nonlocal epoch_len_sec
        nonlocal delta_power_D_episode
        nonlocal bidx_D_episode
        # calculated in the outer function
        nonlocal scale
        nonlocal bidx_valid
        nonlocal boundary_tau_i
        nonlocal boundary_tau_d

        if tau_i<boundary_tau_i[0] or tau_d<boundary_tau_d[0] or tau_i>boundary_tau_i[1] or tau_d>boundary_tau_d[1]:
            # scipy.optimize.brute sometimes tries to search beyond the grid boundary.
            return np.array(np.inf)

        for i, st, size in zip(range(len(episode_stage)), episode_stage, episode_size):
            delta_t = size * epoch_len_sec
            if st == 'I':
                simulated_s[i+1] = _next_S_I(simulated_s[i], delta_t, tau_i)
            else:
                simulated_s[i+1] = _next_S_D(simulated_s[i], delta_t, tau_d)

        # Evaluate the error (sum of the squared differences)
        sim_delta_power_episode = simulated_s[1:]
        sim_delta_power_D_episode = sim_delta_power_episode[bidx_D_episode]
        sim_err = np.sum(
            np.power((scale*sim_delta_power_D_episode[bidx_valid] - scale*delta_power_D_episode[bidx_valid]), 2))

        return np.array(sim_err)

    # These lower boundaries are from Fig.1d of Franken et al. 2001. The upper boundaries are empirical.
    boundary_tau_i = (3600, 30*3600)
    boundary_tau_d = (360, 30*3600)

    # initialize an array for the simulation of process-S
    simulated_s = np.zeros(len(episode_stage) + 1)

    # Pre-calculation of scale required in _evaluate_process_s()
    scale = 1/np.nanmedian(delta_power_D_episode)

    # valid datapoint
    bidx_valid = ~np.isnan(delta_power_D_episode)

    # do the grid search
    opt_taus = optimize.brute(_evaluate_process_s, (boundary_tau_i, boundary_tau_d, (low_asymp, up_asymp)))

    return opt_taus


def empty_median(ar):
    """ calculates median like np.nanmedian() but this returns NaN if the given array is empty (all NaN)
    Arguments:
        ar {np.array} -- an array

    Returns:
        The mean of the array
    """
    if np.all(np.isnan(ar)):
        res = np.nan
    else:
        res = np.nanmedian(ar)

    return (res)


def empty_std(ar):
    """ calculates std like np.nanstd() but this returns NaN if the given array is empty (all NaN)
    Arguments:
        ar {np.array} -- an array
    Returns:
        The standard deviation of the array
    """
    if np.all(np.isnan(ar)):
        res = np.nan
    else:
        res = np.nanstd(ar)

    return (res)


def empty_mean(ar):
    """ calculates mean like np.nanmean() but this returns NaN if the given array is empty (all NaN)
    Arguments:
        ar {np.array} -- an array
    Returns:
        The mean of the array
    """
    if np.all(np.isnan(ar)):
        res = np.nan
    else:
        res = np.nanmean(ar)

    return (res)


def binned_mean(ts, epoch_len_sec):
    """ calculate 60 min binned mean of the given array
    Arguments:
        ts {np.array} -- an array of floats
    Returns:
        np.array of mean of each bin
    """
    tm = ts.reshape(-1, int(3600/epoch_len_sec))  # 60 min (3600 sec) bin

    # We leave an empty slice (a bin of all nans) as nan
    ts_mean = np.apply_along_axis(empty_mean, 1, tm)

    return (ts_mean)


def draw_sim_and_obs_dpd_group_each(sim_ts_mat, obs_ts_mat, mouse_list, epoch_len_sec, output_dir, y_range=None, epoch_range_basal=None):
    """draws the simulation and observed delta-power dynamics for each group in the given list

    Args:
        sim_ts_mat (np.array): A matrix of simulated delta-power dynamics
        obs_ts_mat (np.array): A matrix of observed delta-power dynamics
        mouse_list (list): A list of mouse group from delta_power_dynamics_df['Mouse group]
        epoch_len_sec (int): The time delta of each epoch
        epoch_range_basak (slice): The slice used for selecting the basal epoch range
    """
    ### unique mouse set with the preserved order
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)
    for set_no, mouse_group in enumerate(mouse_set):
        bidx_group = (np.array(mouse_list) == mouse_group)

        obs_ts_binned_mean_mat = np.apply_along_axis(
            binned_mean, 1, obs_ts_mat, epoch_len_sec)
        sim_ts_binned_mean_mat = np.apply_along_axis(
            binned_mean, 1, sim_ts_mat, epoch_len_sec)

        obs_group_ts = obs_ts_binned_mean_mat[bidx_group]
        sim_group_ts = sim_ts_binned_mean_mat[bidx_group]

        # test number:N for sqrt(N) is different depending on time points
        obs_t_num = np.apply_along_axis(
            np.count_nonzero, 0, ~np.isnan(obs_group_ts))
        sim_t_num = np.apply_along_axis(
            np.count_nonzero, 0, ~np.isnan(sim_group_ts))

        # stats time-series
        obs_mean = np.apply_along_axis(empty_mean, 0, obs_group_ts)
        obs_std = np.apply_along_axis(empty_std, 0, obs_group_ts)
        sim_mean = np.apply_along_axis(empty_mean, 0, sim_group_ts)
        sim_std = np.apply_along_axis(empty_std, 0, sim_group_ts)
        obs_sem = obs_std/np.sqrt(obs_t_num)
        sim_sem = sim_std/np.sqrt(sim_t_num)

        # draw
        fig = Figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        num_points = len(obs_mean)
        x = np.arange(0, num_points)

        if y_range is None:
            y_range = [np.nanmin([sim_mean, obs_mean]),
                       np.nanmax([sim_mean, obs_mean])]


        if epoch_range_basal is not None:
            # Marked the extrapolated area with the pale red backgroud color
            ax.axvspan(0, num_points, color='r', alpha=0.1)
            ax.axvspan(epoch_range_basal.start*epoch_len_sec/3600, epoch_range_basal.stop*epoch_len_sec/3600, color='w')

        _set_common_features_delta_power_dynamics(ax, num_points, y_range)

        ax.plot(x, sim_mean, color='r')
        ax.fill_between(x, sim_mean - sim_sem, sim_mean +
                        sim_sem, color='r', alpha=0.3)
        ax.scatter(x, obs_mean, color=stage.COLOR_NREM)
        ax.vlines(x, obs_mean - obs_sem, obs_mean + obs_sem,
                  color=stage.COLOR_NREM, alpha=0.3)


        #  GE for Group Each
        if epoch_range_basal is not None:
            fig.suptitle(
            f'Delta power dynamics each group (extrapolated): {"  ".join([mouse_group])}')

            filename = f'delta-power-dynamics_extrapolated_GE{set_no:02}_{"_".join([mouse_group])}'
            sc.savefig(output_dir, filename, fig)
        else:
            fig.suptitle(
            f'Delta power dynamics each group: {"  ".join([mouse_group])}')

            filename = f'delta-power-dynamics_GE{set_no:02}_{"_".join([mouse_group])}'
            sc.savefig(output_dir, filename, fig)


def draw_sim_dpd_group_comp(sim_ts_mat, mouse_list, epoch_len_sec, output_dir, y_range=None, epoch_range_basal=None):
    """draws the simulation delta-power dynamics for comparison to the control

    Args:
        sim_ts_mat (np.array): A matrix of simulated delta-power dynamics
        mouse_list (list): A list of mouse group from delta_power_dynamics_df['Mouse group]
        epoch_len_sec (int): The time delta of each epoch
        epoch_range_basak (slice): The slice used for selecting the basal epoch range
    """
    ### unique mouse set with the preserved order
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    if len(mouse_set)<2:
        # nothing to do if there's only one group
        return(0)
    else:
        # Use the frist as control, the followings as tests one by one
        bidx_group_ctrl = (np.array(mouse_list) == mouse_set[0])

        sim_ts_binned_mean_mat = np.apply_along_axis(
            binned_mean, 1, sim_ts_mat, epoch_len_sec)
        sim_group_ts_ctrl = sim_ts_binned_mean_mat[bidx_group_ctrl]

        # test number:N for sqrt(N) is different depending on time points
        sim_t_num_ctrl = np.apply_along_axis(
            np.count_nonzero, 0, ~np.isnan(sim_group_ts_ctrl))

        # stats time-series
        sim_mean_ctrl = np.apply_along_axis(empty_mean, 0, sim_group_ts_ctrl)
        sim_std_ctrl = np.apply_along_axis(empty_std, 0, sim_group_ts_ctrl)
        sim_sem_ctrl = sim_std_ctrl/np.sqrt(sim_t_num_ctrl)

        # draw
        for set_no, mouse_group in enumerate(mouse_set[1:]):
            bidx_group_test = (np.array(mouse_list) == mouse_group)
            sim_group_ts_test = sim_ts_binned_mean_mat[bidx_group_test]
            sim_t_num_test = np.apply_along_axis(
                np.count_nonzero, 0, ~np.isnan(sim_group_ts_test))

            sim_mean_test = np.apply_along_axis(
                empty_mean, 0, sim_group_ts_test)
            sim_std_test = np.apply_along_axis(empty_std, 0, sim_group_ts_test)
            sim_sem_test = sim_std_test/np.sqrt(sim_t_num_test)

            fig = Figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            num_points = len(sim_mean_ctrl)
            x = np.arange(0, num_points)

            if y_range is None:
                y_range = [np.nanmin([sim_mean_ctrl, sim_mean_test]), np.nanmax([sim_mean_ctrl, sim_mean_test])]

            if epoch_range_basal is not None:
                # Marked the extrapolated area with the pale red backgroud color
                ax.axvspan(0, num_points, color='r', alpha=0.1)
                ax.axvspan(epoch_range_basal.start*epoch_len_sec/3600, epoch_range_basal.stop*epoch_len_sec/3600, color='w')

            _set_common_features_delta_power_dynamics(ax, num_points, y_range)

            ax.plot(x, sim_mean_ctrl, color='grey')
            ax.fill_between(x, sim_mean_ctrl - sim_sem_ctrl,
                            sim_mean_ctrl + sim_sem_ctrl, color='grey', alpha=0.3)
            ax.plot(x, sim_mean_test, color=COLOR_SERIES[1])
            ax.fill_between(x, sim_mean_test - sim_sem_test, sim_mean_test +
                            sim_sem_test, color=COLOR_SERIES[1], alpha=0.3)

            if epoch_range_basal is None:
                fig.suptitle(
                    f'Delta power dynamics group comparison: {mouse_set[0]} (n={np.sum(bidx_group_ctrl)}) v.s. {mouse_group} (n={np.sum(bidx_group_test)})')

                #  GC for Group Comparison
                filename = f'delta-power-dynamics_GC{set_no:02}_{"_".join([mouse_set[0], mouse_group])}'
                sc.savefig(output_dir, filename, fig)
            else:
                fig.suptitle(
                    f'Delta power dynamics group comparison (extrapolated): {mouse_set[0]} (n={np.sum(bidx_group_ctrl)}) v.s. {mouse_group} (n={np.sum(bidx_group_test)})')

                #  GC for Group Comparison
                filename = f'delta-power-dynamics_extrapolated_GC{set_no:02}_{"_".join([mouse_set[0], mouse_group])}'
                sc.savefig(output_dir, filename, fig)


def draw_2d_plot_of_taus_group_comp(delta_power_dynamics_df, output_dir):
    """ draws 2D scatter plots of tau_i and tau_d (each group to the control)

    Args:
        delta_power_dynamics_df (pd.DataFrame): A data frame prepared in main()
    """
    # mouse set
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    # each test group to the control group
    # control
    bidx_group_ctrl = (delta_power_dynamics_df['Mouse group'] == mouse_set[0])
    tau_i = delta_power_dynamics_df[bidx_group_ctrl]['Tau_i'].to_numpy()
    tau_d = delta_power_dynamics_df[bidx_group_ctrl]['Tau_d'].to_numpy()
    tau_coord_ctrl = np.array([tau_i, tau_d]).T  # observation is in each row

    # error area
    if np.sum(bidx_group_ctrl)>1:
        covar_ctrl = np.cov(tau_coord_ctrl, rowvar=False)
        mean_ctrl = np.mean(tau_coord_ctrl, axis=0)
        w_ctrl, v_ctrl = linalg.eigh(covar_ctrl)
        w_ctrl = 4. * np.sqrt(w_ctrl)  # 95% confidence (2SD) area (2*radius)
        angle_ctrl = np.arctan(v_ctrl[1, 0] / v_ctrl[0, 0])
        angle_ctrl = 180. * angle_ctrl / np.pi  # convert to degrees

    for set_no, mouse_group in enumerate(mouse_set[1:]):
        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel(r'$\tau_{i}$ (hours)', fontsize=18)
        ax.set_ylabel(r'$\tau_{d}$ (hours)', fontsize=18)

        bidx_group = (delta_power_dynamics_df['Mouse group'] == mouse_group)

        tau_i = delta_power_dynamics_df[bidx_group]['Tau_i'].to_numpy()
        tau_d = delta_power_dynamics_df[bidx_group]['Tau_d'].to_numpy()
        tau_coord = np.array([tau_i, tau_d]).T  # observation is in each row

        ax.scatter(tau_coord_ctrl[:, 0], tau_coord_ctrl[:,
                1], color='grey', label=mouse_set[0])
        ax.scatter(tau_coord[:, 0], tau_coord[:, 1],
                color=COLOR_SERIES[1], label=mouse_group)

        # error area (control)
        if np.sum(bidx_group_ctrl)>1:
            ell_ctrl = patches.Ellipse(
            mean_ctrl, w_ctrl[0], w_ctrl[1], 180. + angle_ctrl, facecolor='none', edgecolor='grey')
            ax.add_patch(ell_ctrl)

        # error area (test)
        if np.sum(bidx_group)>1:
            covar = np.cov(tau_coord, rowvar=False)
            mean = np.mean(tau_coord, axis=0)
            w, v = linalg.eigh(covar)
            w = 4. * np.sqrt(w)  # 95% confidence (2SD) area (2*radius)
            angle = np.arctan(v[1, 0] / v[0, 0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = patches.Ellipse(
                mean, w[0], w[1], 180. + angle, facecolor='none', edgecolor=COLOR_SERIES[1])
            ax.add_patch(ell)

        # update limits if necesary
        ax.set_xlim(
            0, max(20, np.max(tau_coord[:, 0]), np.max(tau_coord_ctrl[:, 0])))
        ax.set_ylim(
            0, max(20, np.max(tau_coord[:, 1]), np.max(tau_coord_ctrl[:, 1])))

        fig.suptitle(
            f'Paired group comparison of Taus\n{mouse_set[0]} (n={np.sum(bidx_group_ctrl)}) v.s. {mouse_group} (n={np.sum(bidx_group)})')
        filename = f'delta-power-dynamics_taus_2D-plot_GC{set_no:02}_{"_".join([mouse_set[0], mouse_group])}'
        sc.savefig(output_dir, filename, fig)


def draw_2d_plot_of_taus_all_group(delta_power_dynamics_df, output_dir):
    """ draws 2D scatter plot of tau_i and tau_d (all group in a plot)

    Args:
        delta_power_dynamics_df (pd.DataFrame): A data frame prepared in main()
    """
    # mouse set
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel(r'$\tau_{i}$ (hours)', fontsize=18)
    ax.set_ylabel(r'$\tau_{d}$ (hours)', fontsize=18)

    for set_no, mouse_group in enumerate(mouse_set):
        bidx_group = (delta_power_dynamics_df['Mouse group'] == mouse_group)

        tau_i = delta_power_dynamics_df[bidx_group]['Tau_i'].to_numpy()
        tau_d = delta_power_dynamics_df[bidx_group]['Tau_d'].to_numpy()
        tau_coord = np.array([tau_i, tau_d]).T  # observation is in each row

        ax.scatter(tau_coord[:, 0], tau_coord[:, 1],
                   color=COLOR_SERIES[set_no], label=mouse_group)

        # error area
        if np.sum(bidx_group) > 1:
            covar = np.cov(tau_coord, rowvar=False)
            mean = np.mean(tau_coord, axis=0)
            w, v = linalg.eigh(covar)
            w = 4. * np.sqrt(w)  # 95% confidence (2SD) area (2*radius)
            angle = np.arctan(v[1, 0] / v[0, 0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = patches.Ellipse(
                mean, w[0], w[1], 180. + angle, facecolor='none', edgecolor=COLOR_SERIES[set_no])
            ax.add_patch(ell)

        # update limits if necesary
        ax.set_xlim(0, max(ax.get_xlim()[1], np.max(tau_coord[:, 0])))
        ax.set_ylim(0, max(ax.get_xlim()[1], np.max(tau_coord[:, 1])))

    fig.suptitle(f'All group comparison of Taus')
    ax.legend()
    filename = f'delta-power-dynamics_taus_2D-plot_all-group'
    sc.savefig(output_dir, filename, fig)


def draw_sim_dpd_group_circ_comp(sim_ts_list, delta_power_dynamics_df, epoch_len_sec, output_dir, y_range=None):
    """draws the simulation delta-power dynamics for comparison to the control

    Args:
        sim_ts_list (list): A matrix of simulated delta-power dynamics
        delta_power_dynamics_df (dataframe): A dataframe containing 'Mouse group' list e.g. delta_power_dynamics_df
        epoch_len_sec (int): The time delta of each epoch
    """
    # unique mouse set with the preserved order
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    if len(mouse_set) < 2:
        # nothing to do if there's only one group
        return (0)
    else:
        # Use the frist as control, the followings as tests one by one
        bidx_group_ctrl = (np.array(mouse_list) == mouse_set[0])

        sim_ts_mat = np.array(sim_ts_list)
        sim_ts_binned_mean_mat = np.apply_along_axis(
            binned_mean, 1, sim_ts_mat, epoch_len_sec)
        sim_group_ts_ctrl = sim_ts_binned_mean_mat[bidx_group_ctrl]

        # stats time-series
        sim_mean_ctrl = np.apply_along_axis(
            empty_mean, 0, sim_group_ts_ctrl)

        sim_mean_ctrl_circ = np.reshape(sim_mean_ctrl, [-1, 24])
        tp_num = sim_mean_ctrl_circ.shape[1]
        delta_power_dynamics_circadian_GC_stats_table_df = pd.DataFrame({'Mouse group': np.repeat(mouse_set[0], tp_num), 'Time': np.arange(tp_num),
                                                                         'N': np.repeat(sim_mean_ctrl_circ.shape[0], tp_num),
                                                                         'Mean': np.mean(sim_mean_ctrl_circ, axis=0),
                                                                         'SD': np.std(sim_mean_ctrl_circ, axis=0),
                                                                         'Pvalue': np.repeat(np.nan, tp_num)})
        # draw
        for set_no, mouse_group in enumerate(mouse_set[1:]):
            bidx_group_test = (np.array(mouse_list) == mouse_group)
            sim_group_ts_test = sim_ts_binned_mean_mat[bidx_group_test]

            sim_mean_test = np.apply_along_axis(
                empty_mean, 0, sim_group_ts_test)

            sim_mean_test_circ = np.reshape(sim_mean_test, [-1, 24])

            p_values = np.ones(tp_num)
            for tp in range(tp_num):
                test_res = stats.ttest_ind(
                    sim_mean_ctrl_circ[:, tp], sim_mean_test_circ[:, tp], equal_var=False)
                p_values[tp] = test_res.pvalue

            group_stats = pd.DataFrame({'Mouse group': np.repeat(mouse_group, tp_num), 'Time': np.arange(tp_num),
                                        'N': np.repeat(sim_mean_test_circ.shape[0], tp_num),
                                        'Mean': np.mean(sim_mean_test_circ, axis=0),
                                        'SD': np.std(sim_mean_test_circ, axis=0),
                                        'Pvalue': p_values})
            delta_power_dynamics_circadian_GC_stats_table_df = pd.concat(
                [delta_power_dynamics_circadian_GC_stats_table_df, group_stats])

            fig = Figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            ax.scatter(np.tile(np.arange(0, 24), 2),
                       sim_mean_ctrl, color="grey")
            ax.scatter(np.tile(np.arange(0, 24), 2),
                       sim_mean_test, color=COLOR_SERIES[1])

            if y_range is None:
                y_range = [np.nanmin([sim_mean_ctrl, sim_mean_test]), np.nanmax(
                    [sim_mean_ctrl, sim_mean_test])]

            _set_common_features_delta_power_dynamics(ax, 24, y_range)

            filename = f'delta-power-dynamics_circadian_GC{set_no:02}_{"_".join([mouse_set[0], mouse_group])}'
            sc.savefig(output_dir, filename, fig)

    return delta_power_dynamics_circadian_GC_stats_table_df


def stats_table_of_taus(delta_power_dynamics_df, output_dir):
    """ makes a statistics table of taus and related values

    Args:
        delta_power_dynamics_df (pd.DataFrame): A data frame prepared in main()
    """
    # mouse set
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    delta_power_dynamics_stats_df = pd.DataFrame()

    # mouse_set's index:0 is always control
    mouse_group = mouse_set[0]
    bidx_group = (delta_power_dynamics_df['Mouse group'] == mouse_group)
    group_df = delta_power_dynamics_df[bidx_group]

    num_c = np.sum(bidx_group)
    tau_i_values_c = group_df['Tau_i'].to_numpy(dtype=np.float64)
    tau_d_values_c = group_df['Tau_d'].to_numpy(dtype=np.float64)
    s0_values_c = group_df['S0'].to_numpy(dtype=np.float64)
    low_asymp_values_c = group_df['Lower_asymptote'].to_numpy(dtype=np.float64)
    up_asymp_values_c = group_df['Upper_asymptote'].to_numpy(dtype=np.float64)
    num_of_d_episode_c = group_df['Num_of_D-episode'].to_numpy(
        dtype=np.integer)
    r2_score_c = group_df['R2_score'].to_numpy(dtype=np.float64)

    row1 = [mouse_group, 'Tau_i', num_c, np.mean(
        tau_i_values_c), np.std(tau_i_values_c), np.nan, None, None]
    row2 = [mouse_group, 'Tau_d', num_c, np.mean(
        tau_d_values_c), np.std(tau_d_values_c), np.nan, None, None]
    row3 = [mouse_group, 'S0', num_c, np.mean(
        s0_values_c), np.std(s0_values_c), np.nan, None, None]
    row4 = [mouse_group, 'Lower asymptote', num_c, np.mean(
        low_asymp_values_c), np.std(low_asymp_values_c), np.nan, None, None]
    row5 = [mouse_group, 'Upper asymptote', num_c, np.mean(
        up_asymp_values_c), np.std(up_asymp_values_c), np.nan, None, None]
    row6 = [mouse_group, 'Number of D-episode', num_c,
            np.mean(num_of_d_episode_c), np.std(num_of_d_episode_c), np.nan, None, None]
    row7 = [mouse_group, 'R2 Score', num_c, np.mean(
        r2_score_c), np.std(r2_score_c), np.nan, None, None]

    new_rows = pd.DataFrame([row1, row2, row3, row4, row5, row6, row7])
    delta_power_dynamics_stats_df = pd.concat(
        [delta_power_dynamics_stats_df, new_rows], ignore_index=True)

    for mouse_group in mouse_set[1:]:
        bidx_group = (delta_power_dynamics_df['Mouse group'] == mouse_group)
        group_df = delta_power_dynamics_df[bidx_group]

        num_t = np.sum(bidx_group)
        tau_i_values_t = group_df['Tau_i'].to_numpy(dtype=np.float64)
        tau_d_values_t = group_df['Tau_d'].to_numpy(dtype=np.float64)
        s0_values_t = group_df['S0'].to_numpy(dtype=np.float64)
        low_asymp_values_t = group_df['Lower_asymptote'].to_numpy(
            dtype=np.float64)
        up_asymp_values_t = group_df['Upper_asymptote'].to_numpy(
            dtype=np.float64)
        num_of_d_episode_t = group_df['Num_of_D-episode'].to_numpy(
            dtype=np.integer)
        r2_score_t = group_df['R2_score'].to_numpy(dtype=np.float64)

        t1 = sc.test_two_sample(tau_i_values_c, tau_i_values_t)
        t2 = sc.test_two_sample(tau_d_values_c, tau_d_values_t)
        t3 = sc.test_two_sample(s0_values_c, s0_values_t)
        t4 = sc.test_two_sample(low_asymp_values_c, low_asymp_values_t)
        t5 = sc.test_two_sample(up_asymp_values_c, up_asymp_values_t)
        t6 = sc.test_two_sample(num_of_d_episode_c, num_of_d_episode_t)
        t7 = sc.test_two_sample(r2_score_c, r2_score_t)

        row1 = [mouse_group, 'Tau_i', num_t, np.mean(tau_i_values_t), np.std(
                tau_i_values_t), t1['p_value'], t1['stars'], t1['method']]
        row2 = [mouse_group, 'Tau_d', num_t, np.mean(tau_d_values_t), np.std(
                tau_d_values_t), t2['p_value'], t2['stars'], t2['method']]
        row3 = [mouse_group, 'S0', num_t, np.mean(s0_values_t), np.std(
                s0_values_t), t3['p_value'], t3['stars'], t3['method']]
        row4 = [mouse_group, 'Lower asymptote', num_t, np.mean(low_asymp_values_t), np.std(
                low_asymp_values_t), t4['p_value'], t4['stars'], t4['method']]
        row5 = [mouse_group, 'Upper asymptote', num_t, np.mean(up_asymp_values_t), np.std(
                up_asymp_values_t), t5['p_value'], t5['stars'], t5['method']]
        row6 = [mouse_group, 'Number of D-episode', num_t, np.mean(num_of_d_episode_t), np.std(
                num_of_d_episode_t), t6['p_value'], t6['stars'], t6['method']]
        row7 = [mouse_group, 'R2 Score', num_t, np.mean(r2_score_t), np.std(
                r2_score_t), t7['p_value'], t7['stars'], t7['method']]

        new_rows = pd.DataFrame([row1, row2, row3, row4, row5, row6, row7])
        delta_power_dynamics_stats_df = pd.concat(
            [delta_power_dynamics_stats_df, new_rows], ignore_index=True)

    delta_power_dynamics_stats_df.columns = ['Mouse group', 'Variable',
                                             'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    delta_power_dynamics_df.to_csv(os.path.join(
        output_dir, 'delta-power-dynamics_table.csv'), index=False)
    delta_power_dynamics_stats_df.to_csv(os.path.join(
        output_dir, 'delta-power-dynamics_stats_table.csv'), index=False)


def draw_barchart_of_taus_group_comp(delta_power_dynamics_df, output_dir):
    # mouse set
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    mouse_groups_set = sorted(set(mouse_list), key=mouse_list.index)
    num_groups = len(mouse_groups_set)

    if num_groups > 1:
        bidx_group_ctrl = delta_power_dynamics_df['Mouse group'] == mouse_groups_set[0]
        # Tau_i (control)
        values_ti_c = delta_power_dynamics_df['Tau_i'].values[bidx_group_ctrl]
        mean_ti_c = np.mean(values_ti_c)
        sem_ti_c = np.std(values_ti_c)/np.sqrt(len(values_ti_c))

        # Tau_d (control)
        values_td_c = delta_power_dynamics_df['Tau_d'].values[bidx_group_ctrl]
        mean_td_c = np.mean(values_td_c)
        sem_td_c = np.std(values_td_c)/np.sqrt(len(values_td_c))

        for g_idx, mouse_group in enumerate(mouse_groups_set[1:]):
            w = 0.8  # bar width
            xtick_str_list = ['\n'.join(textwrap.wrap(mouse_groups_set[0], 8)),
                              '\n'.join(textwrap.wrap(mouse_group, 8))]
            x_pos = [0, 1]
            fig = Figure(figsize=(7, 4))
            fig.subplots_adjust(wspace=0.5)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.set_xticks(x_pos)
            ax2.set_xticks(x_pos)
            ax1.set_xticklabels(xtick_str_list)
            ax2.set_xticklabels(xtick_str_list)
            ax1.set_ylabel(r'$\tau_{i}$ (hours)', fontsize=14)
            ax2.set_ylabel(r'$\tau_{d}$ (hours)', fontsize=14)

            bidx_group = delta_power_dynamics_df['Mouse group'] == mouse_group

            # Tau_i
            ax1.bar(0, mean_ti_c, yerr=sem_ti_c, align='center',
                    width=w, capsize=6, color=COLOR_SERIES[0], alpha=0.6)
            summary.scatter_datapoints(ax1, w, 0, values_ti_c)

            values_t = delta_power_dynamics_df['Tau_i'].values[bidx_group]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax1.bar(1, mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=COLOR_SERIES[1], alpha=0.6)
            summary.scatter_datapoints(ax1, w, 1, values_t)

            # Tau_d
            ax2.bar(0, mean_td_c, yerr=sem_td_c, align='center',
                    width=w, capsize=6, color=COLOR_SERIES[0], alpha=0.6)
            summary.scatter_datapoints(ax2, w, 0, values_td_c)

            values_t = delta_power_dynamics_df['Tau_d'].values[bidx_group]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax2.bar(1, mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=COLOR_SERIES[1], alpha=0.6)
            summary.scatter_datapoints(ax2, w, 1, values_t)
            fig.suptitle(
                f'Paired group comparison of Taus \n{mouse_groups_set[0]} (n={len(values_ti_c)}) v.s. {mouse_group} (n={len(values_t)})')

            filename = f'delta-power-dynamics_taus_barchart_GC{g_idx:02}_{"_".join([mouse_groups_set[0], mouse_group])}'
            sc.savefig(output_dir, filename, fig)

    else:
        # Nothing to do
        return (0)


def draw_barchart_of_taus_all_group(delta_power_dynamics_df, output_dir):
    # mouse set
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    mouse_groups_set = sorted(set(mouse_list), key=mouse_list.index)
    num_groups = len(mouse_groups_set)

    bidx_group_list = [delta_power_dynamics_df['Mouse group'] == g for g in mouse_groups_set]

    fig = Figure(figsize=(7, 4))
    fig.subplots_adjust(wspace=0.5)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    w = 0.8  # bar width
    x_pos = range(num_groups)
    xtick_str_list = ['\n'.join(textwrap.wrap(mouse_groups_set[g_idx], 8))
                    for g_idx in range(num_groups)]

    ax1.set_xticks(x_pos)
    ax2.set_xticks(x_pos)
    ax1.set_xticklabels(xtick_str_list)
    ax2.set_xticklabels(xtick_str_list)
    ax1.set_ylabel(r'$\tau_{i}$ (hours)', fontsize=14)
    ax2.set_ylabel(r'$\tau_{d}$ (hours)', fontsize=14)

    if num_groups > 1:
        # Tau_i
        values_c = delta_power_dynamics_df['Tau_i'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax1.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color=COLOR_SERIES[0], alpha=0.6)
        summary.scatter_datapoints(ax1, w, x_pos[0], values_c)
        for g_idx in range(1, num_groups):
            values_t = delta_power_dynamics_df['Tau_i'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=COLOR_SERIES[g_idx], alpha=0.6)
            summary.scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        # Tau_d
        values_c = delta_power_dynamics_df['Tau_d'].values[bidx_group_list[0]]
        mean_c = np.mean(values_c)
        sem_c = np.std(values_c)/np.sqrt(len(values_c))
        ax2.bar(x_pos[0], mean_c, yerr=sem_c, align='center',
                width=w, capsize=6, color=COLOR_SERIES[0], alpha=0.6)
        summary.scatter_datapoints(ax2, w, x_pos[0], values_c)

        for g_idx in range(1, num_groups):
            values_t = delta_power_dynamics_df['Tau_d'].values[bidx_group_list[g_idx]]
            mean_t = np.mean(values_t)
            sem_t = np.std(values_t)/np.sqrt(len(values_t))
            ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                    width=w, capsize=6, color=COLOR_SERIES[g_idx], alpha=0.6)
            summary.scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

    else:
        # single group
        g_idx = 0
        # Tau_i
        values_t = delta_power_dynamics_df['Tau_i'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax1.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=COLOR_SERIES[1], alpha=0.6)
        summary.scatter_datapoints(ax1, w, x_pos[g_idx], values_t)

        # Tau_d
        values_t = delta_power_dynamics_df['Tau_d'].values[bidx_group_list[0]]
        mean_t = np.mean(values_t)
        sem_t = np.std(values_t)/np.sqrt(len(values_t))
        ax2.bar(x_pos[g_idx], mean_t, yerr=sem_t, align='center',
                width=w, capsize=6, color=COLOR_SERIES[1], alpha=0.6)
        summary.scatter_datapoints(ax2, w, x_pos[g_idx], values_t)

    fig.suptitle(r'All group comparison of Taus')

    filename = f'delta-power-dynamics_taus_barchart_all-group'
    sc.savefig(output_dir, filename, fig)


def draw_boxplot_of_asymptotes(asymptote_df, output_dir):

    # mouse set
    mouse_list = asymptote_df['Mouse group'].tolist()
    mouse_groups_set = sorted(set(mouse_list), key=mouse_list.index)
    num_groups = len(mouse_groups_set)
    mouse_groups_set

    # the control group
    bidx_group_ctrl = asymptote_df['Mouse group'] == mouse_groups_set[0]

    # Lower & Upper asymptote (control)
    values_low_c = asymptote_df['Lower_asymptote'].values[bidx_group_ctrl]
    values_up_c = asymptote_df['Upper_asymptote'].values[bidx_group_ctrl]
    num_list = [np.sum(bidx_group_ctrl)]
    asymp_values_list = [values_low_c, values_up_c]
    xtick_str_list = ['\n'.join(textwrap.wrap(mouse_groups_set[0]+' Low', 8)),
                      '\n'.join(textwrap.wrap(mouse_groups_set[0]+' Up', 8))]

    # test groups
    for g_idx, mouse_group in enumerate(mouse_groups_set[1:]):

        bidx_group = asymptote_df['Mouse group'] == mouse_group
        num_list.extend([np.sum(bidx_group)])

        # Lower asymptote
        values_t = asymptote_df['Lower_asymptote'].values[bidx_group]
        asymp_values_list.append(values_t)

        # Upper asymptote
        values_t = asymptote_df['Upper_asymptote'].values[bidx_group]
        asymp_values_list.append(values_t)

        # tick labels
        xtick_str_list.extend(['\n'.join(textwrap.wrap(mouse_groups_set[g_idx+1] + ' Low', 8)),
                               '\n'.join(textwrap.wrap(mouse_groups_set[g_idx+1] + ' Up', 8))])

    # draw the figure
    fig = Figure(figsize=(7, 4))
    fig.subplots_adjust(wspace=0.5)
    ax = fig.add_subplot(111)
    w = 0.8  # bar width

    bp = ax.boxplot(asymp_values_list, widths=w, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_alpha(0.6)
        box.set_facecolor(COLOR_SERIES[int(i/2)])
        box.set_zorder(1)

    for box in bp['medians']:
        box.set(color='black', linewidth=1)

    for x_pos, vals in enumerate(asymp_values_list):
        summary.scatter_datapoints(ax, w, x_pos + 1, vals)

    ax.set_ylabel(r'Delta power [%]', fontsize=14)
    ax.set_xticklabels(xtick_str_list)
    ax.set_axisbelow(True)
    ax.grid(axis='y')

    fig.suptitle(
        f'Comparison of asymptotes \n'
        f'{", ".join([mouse_group +"(n="+str(num)+")" for mouse_group, num in zip(mouse_groups_set, num_list)])}')
    filename = f'delta-power-dynamics_comparison_of_asymptotes'
    sc.savefig(output_dir, filename, fig)


def make_asymptote_df(mouse_info_df, stage_ext, epoch_range_basal, csv_head, csv_body, output_dir):
    """makes a dataframe of upper and lower asymptotes

    Args:
        mouse_info_df (pd.DataFrame): The dataframe of mouse_info

    Returns:
        asymptote_df (pd.DataFrame)
    """

    asymptote_df = pd.DataFrame(columns=['Experiment label', 'Mouse group', 'Mouse ID', 'Device label',
                                                        'Lower_asymptote', 'Upper_asymptote'])

    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label'].strip()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        stats_report = r['Stats report'].strip().upper()
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']
        if stats_report == 'NO':
            print_log(f'[{i+1}] Skipping: {faster_dir} {device_label}')
            continue
        print_log(
            f'[{i+1}] Estimating asymptotes: {faster_dir} {device_label} {stage_ext}')

        # read the stage file
        stage_call_all = et.read_stages(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)
        stage_call = stage_call_all[epoch_range_basal]

        # read the delta power timeseries
        keys = {'Experiment label': exp_label, 'Mouse group': mouse_group,
                'Mouse ID': mouse_id, 'Device label': device_label}
        delta_power_all = select_delta_power(csv_head, csv_body, keys)
        delta_power = delta_power_all[epoch_range_basal]


        # Estimate the lower and upper asymptotes
        (low_asymp, up_asymp, fig) = estimate_delta_power_asymptote(
            stage_call, delta_power)
        fig.suptitle(
            f'Upper and lower asymptotes:\n {"  ".join([exp_label, mouse_group, mouse_id, device_label])}')
        filename = f'asymptotes_estimation_{"_".join([exp_label, mouse_group, mouse_id, device_label])}'
        sc.savefig(output_dir, filename, fig)

        a_row = pd.DataFrame([{'Experiment label': exp_label, 'Mouse group': mouse_group, 'Mouse ID': mouse_id, 'Device label': device_label,
                            'Lower_asymptote': low_asymp, 'Upper_asymptote': up_asymp}])

        asymptote_df = pd.concat([asymptote_df, a_row])

    return asymptote_df


def do_analysis(mouse_info_df, stage_ext, epoch_range_basal, csv_body, csv_head, epoch_len_sec, output_dir, bool_extrapolation):
    """ Calculates Ti and Td (the main part of the delta-power dynamics analyssi)

    Args:
        mouse_info_df (pd.DataFrame): contents of collected_mouse_info_df.json (mouse.info.csv)
        stage_ext (str): The extention for the stage files
        epoch_range_basal (slice): The range of the epochs used as the basal
        csv_body (pd.DataFrame): dataframe returned by read_delta_power_csv()
        csv_head (pd.DataFrame): dataframe returned by read_delta_power_csv()
        epoch_len_sec (int): The length [sec] of each epoch
        output_dir (str): The folder path of the output files
        bool_extrapolation (bool): The extrapolated plots are generated when true

    Returns:
        sim_ts_list: simulated time-series of delta-power
        obs_ts_list: observed time-series of delta-power
        sim_ts_ext_list: simulated time-series of delta-power (extrapolated)
        obs_ts_ext_list: observed time-series of delta-power (extrapolated)
        delta_power_dynamics_df: Summary table of the delta-power dynamics
    """
    # Estimate the asymptotes
    asymptote_df = make_asymptote_df(mouse_info_df, stage_ext, epoch_range_basal, csv_head, csv_body, output_dir)
    asymptote_medians_df = asymptote_df.groupby('Mouse group').median()

    ## Initialize dataframe and lists to store results
    delta_power_dynamics_df = pd.DataFrame(columns=['Experiment label', 'Mouse group', 'Mouse ID', 'Device label',
                                                    'Lower_asymptote', 'Upper_asymptote',
                                                    'Tau_i', 'Tau_d', 'S0', 'Num_of_D-episode', 'R2_score'])
    sim_ts_list = list()  # a list of the simulated delta power dynamics time-series
    obs_ts_list = list()  # a list of the observed delta power dynamics time-series
    sim_ts_ext_list = list() # a list of the simulated delta power dynamics time-series (extrapolation)
    obs_ts_ext_list = list() # a list of the observed delta power dynamics time-series (extrapolation)

    ## Optimize parameters for each mouse and simulate the dynamics
    for i, r in mouse_info_df.iterrows():
        device_label = r['Device label'].strip()
        mouse_group = r['Mouse group'].strip()
        mouse_id = r['Mouse ID'].strip()
        stats_report = r['Stats report'].strip().upper()
        exp_label = r['Experiment label'].strip()
        faster_dir = r['FASTER_DIR']
        if stats_report == 'NO':
            print_log(f'[{i+1}] Skipping: {faster_dir} {device_label}')
            continue
        print_log(
            f'[{i+1}] Reading stage & delta power: {faster_dir} {device_label} {stage_ext}')

        # read the stage file
        stage_call_all = et.read_stages(os.path.join(
            faster_dir, 'result'), device_label, stage_ext)
        stage_call = stage_call_all[epoch_range_basal]

        # read the delta power timeseries
        keys = {'Experiment label': exp_label, 'Mouse group': mouse_group,
                'Mouse ID': mouse_id, 'Device label': device_label}
        delta_power_all = select_delta_power(csv_head, csv_body, keys)
        delta_power = delta_power_all[epoch_range_basal]

        if bool_extrapolation and (len(delta_power_all) == len(delta_power)):
            print_log('[Warning] The extrapolation was requested but canceled because the summary has only the basal range of epochs.')
            bool_extrapolation = False


        # make episodes of 'I'ncreasing stages and 'D'ecreasing stages
        episode_len = int(np.ceil(EPISODE_LEN_MINUTES*60/epoch_len_sec))
        (episode_stage, episode_size, idx_episode_start) = make_episode(
            stage_call, episode_len)

        # make D-episodes
        (num_of_D_episode, delta_power_D_episode, idx_D_episode, bidx_D_episode) = make_D_episode(delta_power,
                                                                                                  episode_size, episode_len,
                                                                                                  episode_stage, idx_episode_start)

        # set asymptotes
        low_asymp_group = asymptote_medians_df.loc[mouse_group]['Lower_asymptote']
        up_asymp_group = asymptote_medians_df.loc[mouse_group]['Upper_asymptote']


        # Do fitting to find the optimal parameters (Taus and S0)
        print_log(
            f'    Optimizing the simulation of delta power dynamics for {num_of_D_episode} D-episodes')
        opt_taus = do_fitting(episode_stage, episode_size, bidx_D_episode,
                              delta_power_D_episode, epoch_len_sec, low_asymp_group, up_asymp_group)

        # Do the simulation based on the found paramter
        sim_ts, obs_ts = simulate_delta_power_dynamics(
            opt_taus, low_asymp_group, up_asymp_group, stage_call, idx_D_episode, delta_power_D_episode, epoch_len_sec)

        # Make a row for the stats table
        bidx_valid_obs = ~np.isnan(obs_ts)
        a_row = pd.DataFrame([{'Experiment label': exp_label, 'Mouse group': mouse_group, 'Mouse ID': mouse_id, 'Device label': device_label,
                               'Lower_asymptote': low_asymp_group, 'Upper_asymptote': up_asymp_group,
                               'Tau_i': opt_taus[0]/3600, 'Tau_d':opt_taus[1]/3600, 'S0':opt_taus[2], 'Num_of_D-episode':num_of_D_episode,
                               'R2_score':r2_score(obs_ts[bidx_valid_obs], sim_ts[bidx_valid_obs])}])

        delta_power_dynamics_df = pd.concat([delta_power_dynamics_df, a_row])
        sim_ts_list.append(sim_ts)
        obs_ts_list.append(obs_ts)

        # Do the extrapolated simulation based on the found paramter
        if bool_extrapolation:
            # make episode with all epochs
            (episode_stage, episode_size, idx_episode_start) = make_episode(
                stage_call_all, episode_len)
            (num_of_D_episode, delta_power_D_episode, idx_D_episode, bidx_D_episode) = make_D_episode(delta_power_all,
                                                                                                    episode_size, episode_len,
                                                                                                    episode_stage, idx_episode_start)
            # Do simulation based on the found paramter
            sim_ts, obs_ts = simulate_delta_power_dynamics(
                opt_taus, low_asymp_group, up_asymp_group, stage_call_all, idx_D_episode, delta_power_D_episode, epoch_len_sec)

            sim_ts_ext_list.append(sim_ts)
            obs_ts_ext_list.append(obs_ts)

    delta_power_dynamics_df.index = np.arange(len(delta_power_dynamics_df))

    return (sim_ts_list, obs_ts_list, sim_ts_ext_list, obs_ts_ext_list, delta_power_dynamics_df, asymptote_df)


def draw_plots(delta_power_dynamics_df, asymptote_df, sim_ts_list, obs_ts_list, sim_ts_ext_list, obs_ts_ext_list, epoch_len_sec, epoch_range_basal, output_dir, bool_extrapolation):
    """ make plots of
        1) simulated_delta_power_dynamics with observed data points for each animal,
        2) simulated_delta_power_dynamics with observed data points for each group,
        3) group comparison of the simulated delta power dynamics,
        4) 2D plots of Ti and Td,
        5) Barchart ot Ti and Td.

    Args:
        delta_power_dynamics_df (pd.DataFrame): a DataFrame made by main_process()
        sim_ts_list (2D list): simulated timeseries of mice
        obs_ts_list (2D list): observed timeseries of mice
        sim_ts_ext_list (2D list): simulated timeseries (extrapolated) of mice
        obs_ts_ext_list (2D list): observed timeseries (extrapolated) of mice
        epoch_len_sec (int): The lengch of an epoch in seconds
        epoch_range_basal (tuple or list): Epoch range of the basal
        bool_extrapolation (bool): The boolean switch to direct whether to do the extrapolation or not
    """
    if bool_extrapolation:
        y_min = np.min([np.min(sim_ts_list), np.nanmin(obs_ts_list), np.min(sim_ts_ext_list), np.nanmin(obs_ts_ext_list)])
        y_max = np.max([np.max(sim_ts_list), np.nanmax(obs_ts_list), np.max(sim_ts_ext_list), np.nanmax(obs_ts_ext_list)])
    else:
        y_min = np.min([np.min(sim_ts_list), np.nanmin(obs_ts_list)])
        y_max = np.max([np.max(sim_ts_list), np.nanmax(obs_ts_list)])

    # Draw plots
    for sim_ts, obs_ts, exp_label, mouse_group, mouse_id, device_label in zip(sim_ts_list, obs_ts_list,
                          delta_power_dynamics_df['Experiment label'],
                          delta_power_dynamics_df['Mouse group'],
                          delta_power_dynamics_df['Mouse ID'],
                          delta_power_dynamics_df['Device label']):
        fig = draw_simulated_delta_power_dynamics(
            sim_ts, obs_ts, epoch_len_sec, [y_min, y_max])
        fig.suptitle(
            f'Delta power dynamics: {"  ".join([exp_label, mouse_group, mouse_id, device_label])}')
        filename = f'delta-power-dynamics_{"_".join([exp_label, mouse_group, mouse_id, device_label])}'
        sc.savefig(output_dir, filename, fig)

    if bool_extrapolation:
        for sim_ts, obs_ts, exp_label, mouse_group, mouse_id, device_label in zip(sim_ts_ext_list, obs_ts_ext_list,
                          delta_power_dynamics_df['Experiment label'],
                          delta_power_dynamics_df['Mouse group'],
                          delta_power_dynamics_df['Mouse ID'],
                          delta_power_dynamics_df['Device label']):
            fig = draw_simulated_delta_power_dynamics(sim_ts, obs_ts, epoch_len_sec, [y_min, y_max], epoch_range_basal)
            fig.suptitle(
                f'Delta power dynamics (extrapolated): {"  ".join([exp_label, mouse_group, mouse_id, device_label])}')
            filename = f'delta-power-dynamics_extrapolated_{"_".join([exp_label, mouse_group, mouse_id, device_label])}'
            sc.savefig(output_dir, filename, fig)



    ## Superimposed plot of simulation on the observed delta-power
    mouse_list = delta_power_dynamics_df['Mouse group'].tolist()
    obs_ts_mat = np.array(obs_ts_list)
    sim_ts_mat = np.array(sim_ts_list)
    draw_sim_and_obs_dpd_group_each(sim_ts_mat, obs_ts_mat, mouse_list, epoch_len_sec, output_dir, [y_min, y_max])
    draw_sim_dpd_group_comp(sim_ts_mat, mouse_list, epoch_len_sec, output_dir, [y_min, y_max])

    if bool_extrapolation:
        obs_ts_ext_mat = np.array(obs_ts_ext_list)
        sim_ts_ext_mat = np.array(sim_ts_ext_list)
        draw_sim_and_obs_dpd_group_each(sim_ts_ext_mat, obs_ts_ext_mat, mouse_list, epoch_len_sec, output_dir, [y_min, y_max], epoch_range_basal)
        draw_sim_dpd_group_comp(sim_ts_ext_mat, mouse_list, epoch_len_sec, output_dir, [y_min, y_max], epoch_range_basal)

    # Draw
    draw_boxplot_of_asymptotes(asymptote_df, output_dir)

    # Draw 2D plots of Tau_i and Tau_d
    draw_2d_plot_of_taus_group_comp(delta_power_dynamics_df, output_dir)
    draw_2d_plot_of_taus_all_group(delta_power_dynamics_df, output_dir)

    # Make a statistics table and draw bar charts
    stats_table_of_taus(delta_power_dynamics_df, output_dir)
    draw_barchart_of_taus_group_comp(delta_power_dynamics_df, output_dir)
    draw_barchart_of_taus_all_group(delta_power_dynamics_df, output_dir)

    # Draw the comparison plot of circadian profiles
    circadian_GC_stats_table_df = draw_sim_dpd_group_circ_comp(sim_ts_list, delta_power_dynamics_df, epoch_len_sec, output_dir)
    circadian_GC_stats_table_df.to_csv(os.path.join(
        output_dir, 'delta-power-dynamics_circadian_GC_stats_table.csv'), index=False)


def main(args, summary_dir, output_dir):
    """ is a main process

    Args:
        args (dict): A dict given by the PARSER.parse_args()
    """

    # pylint: disable = invalid-name, global-variable-not-assigned
    global log

    # read the collected mouse information used for the summary
    try:
        with open(os.path.join(summary_dir, 'collected_mouse_info_df.json'), 'r',
                  encoding='UTF-8') as infile:
            mouse_info_collected = json.load(infile)
    except FileNotFoundError as err:
        print_log(
            f'Failed to find collected_mouse_info_df.json. Check the summary folder path is valid. {err}')
        return
    mouse_info_collected['mouse_info'] = pd.read_json(
        mouse_info_collected['mouse_info'], orient="table")

    # basic parameters of the summary
    mouse_info_df = mouse_info_collected['mouse_info']
    epoch_num = mouse_info_collected['epoch_num']
    epoch_len_sec = mouse_info_collected['epoch_len_sec']
    stage_ext = mouse_info_collected['stage_ext']

    if stage_ext is None:
        stage_ext = 'faster2'

    # set the epoch range to be summarized
    if args.epoch_range:
        # use the range given by the command line option
        e_range = [
            int(x.strip()) if x else None for x in args.epoch_range.split(':')]
        epoch_range_basal = slice(*e_range)
    else:
        # default: use the all epochs
        epoch_range_basal = slice(0, epoch_num, None)

    # The request of the extrapolation may be cancelled later
    bool_extrapolation = args.extrapolation

    # read delta-power CSV file
    path_to_delta_power_csv = os.path.join(
        summary_dir, 'PSD_norm', 'power-timeseries_norm_delta_percentage.csv')
    try:
        (csv_head, csv_body) = read_delta_power_csv(path_to_delta_power_csv)
    except FileNotFoundError as err:
        print_log(
            f'Failed to find the CSV file. Check the summary folder path is valid. {err}')

    # Main process

    if (epoch_range_basal.stop - epoch_range_basal.start) > epoch_num:
        print_log(f'[Error] The specified epoch range:{epoch_range_basal.start}-{epoch_range_basal.stop} is out of the index ({epoch_num}).')
        return -1

    ## print log the basic information
    print_log(f'The basal-epoch range: {epoch_range_basal.start}-{epoch_range_basal.stop} '\
              f'({epoch_range_basal.stop - epoch_range_basal.start} epochs out of {epoch_num}). '\
              f'Extrapolation: {args.extrapolation}')

    (sim_ts_list, obs_ts_list, sim_ts_ext_list, obs_ts_ext_list, delta_power_dynamics_df, asymptote_df) = do_analysis(
        mouse_info_df, stage_ext, epoch_range_basal, csv_body, csv_head, epoch_len_sec, output_dir, bool_extrapolation)

    draw_plots(delta_power_dynamics_df, asymptote_df, sim_ts_list, obs_ts_list, sim_ts_ext_list,
               obs_ts_ext_list, epoch_len_sec, epoch_range_basal, output_dir, bool_extrapolation)

if __name__ == '__main__':
    # initialize global variables
    # pylint: disable = invalid-name
    log = None

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-s", "--summary_dir", required=True,
                        help="a path to the summary folder")
    PARSER.add_argument("-e", "--epoch_range",
                        help="the range of basal epochs to be analyzed (default: '0:epoch_num'")
    PARSER.add_argument("-x", "--extrapolation",
                        help="flag to extrapolate the simulation to the whole epochs"
                        "(default: false)",
                        action='store_true')

    args = PARSER.parse_args()

    summary_dir = os.path.abspath(args.summary_dir)
    if not os.path.exists(summary_dir):
        print(f'Failed to open the summary folder: {summary_dir}')
        sys.exit(1)

    output_dir = os.path.join(summary_dir, 'delta_power_dynamics')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pdf'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)

    # initialize logger
    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(
        output_dir, 'log', f'summary.{dt_str}.log'))
    print_log(f'[{dt_str} - {stage.FASTER2_NAME} - {sys.modules[__name__].__file__}]'
              f' Started in: {os.path.abspath(output_dir)}')

    try:
        main(args, summary_dir, output_dir)
    # pylint: disable = broad-except
    except Exception as e:
        print_log_exception('Unhandled exception occured')
