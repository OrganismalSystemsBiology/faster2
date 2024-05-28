# -*- coding: utf-8 -*-
import os
# Limit the number of BLAS threads because GMM also tries multithreading
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['GOTO_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
# pylint: disable = wrong-import-position
import sys
import argparse
import pandas as pd
import numpy as np
import faster2lib.eeg_tools as et
import re
from datetime import datetime, timedelta
from scipy import signal
from hmmlearn import hmm, base
from sklearn import mixture
import matplotlib as mpl
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from scipy import linalg, stats
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import pickle
from glob import glob
import mne
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter
import traceback


FASTER2_NAME = 'FASTER2 version 0.4.8'
STAGE_LABELS = ['Wake', 'REM', 'NREM']
XLABEL = 'Total low-freq. log-powers'
YLABEL = 'Total high-freq. log-powers'
ZLABEL = 'REM metric'
SCATTER_PLOT_FIG_WIDTH = 6   # inch
SCATTER_PLOT_FIG_HEIGHT = 6  # inch
FIG_DPI = 100  # dot per inch
COLOR_WAKE = '#DC267F'
COLOR_NREM = '#648FFF'
COLOR_REM = '#FFB000'
COLOR_LIGHT = '#FFD700'  # 'gold'
COLOR_DARK = '#696969'  # 'dimgray'
COLOR_DARKLIGHT = 'lightgray'  # light hours in DD condition


class CustomedGHMM(hmm.GaussianHMM):
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        super().__init__(n_components, covariance_type,
                         min_covar, startprob_prior, transmat_prior,
                         means_prior, means_weight,
                         covars_prior, covars_weight,
                         algorithm, random_state,
                         n_iter, tol, verbose,
                         params, init_params)
        self.wr_boundary = None
        self.nr_boundary = None
        self.max_rem_ax = None

    def set_wr_boundary(self, wr_boundary):
        # Wake/REM boundary. REM cluster cannot grow below this boundary
        self.wr_boundary = wr_boundary

    def set_nr_boundary(self, nr_boundary):
        # NREM/REM boundary. REM cluster cannot grow beyond this boundary
        self.nr_boundary = nr_boundary

    def set_max_rem_ax(self, max_rem_ax_len):
        # maximum length of REM's principal axis
        self.max_rem_ax = max_rem_ax_len

    def _confine_REM_in_boundary(self, rem_mean, rem_cov):
        """ By definition, REM cluster is not likely z (i.e. REM-metric)<REM_floor and
        x (i.e. low-freq power)>0.
        This function focuses on the ellipsoid that represents the 95% confidence area
        of REM cluster. If this function finds any principal axis of the ellipsoid
        penetrating the REM floor or the NREM wall (i.e. the end-point of the principal
        axis at z<REM_floor or x>0), it shrinks the length of the axis to the point on
        the constraints.
        """

        w, v = linalg.eigh(rem_cov)
        # all eigenvalues must be positive
        if np.any(w <= 0):
            raise ValueError('Invalid_REM_Cluster')

        w = 2 * np.sqrt(w)  # 95% confidence (2SD) area

        # confine above REM floor
        prn_ax = v@np.diag(w)  # 3x3 matrix: each column is the principal axis
        for i in range(3):
            arr_hd = rem_mean + prn_ax[:, i]  # the arrow head from the mean
            # the negative arrow head from the mean
            narr_hd = rem_mean - prn_ax[:, i]
            if arr_hd[2] < self.wr_boundary:
                sr = (rem_mean[2] - self.wr_boundary) / \
                    (rem_mean[2] - arr_hd[2])  # shrink ratio
            elif narr_hd[2] < self.wr_boundary:
                sr = (rem_mean[2] - self.wr_boundary)/(rem_mean[2] - narr_hd[2])
            else:
                sr = 1
            w[i] = w[i] * sr
        
        # confine the REM cluster within negative low-freq and above the diagonal line
        prn_ax = v@np.diag(w)  # 3x3 matrix: each column is the principal axis
        for i in range(3):
            arr_hd = rem_mean + prn_ax[:, i] # the arrow head from the mean
            narr_hd = rem_mean - prn_ax[:, i] # the negative arrow head from the mean
            
            # condition 1: negative low-freq and BELOW the diagonal line
            if arr_hd[0] > self.nr_boundary and arr_hd[1] < arr_hd[0]:
                # condition 2: if positive high-freq > 0 then allow to grow onto the diagonal line
                if arr_hd[1] > 0:
                    sr = self._shrink_ratio(arr_hd, rem_mean)
                # Otherwise (negative high-freq) then only allow to reach onto the y-axis. 
                else: 
                    sr = (self.nr_boundary - rem_mean[0])/(arr_hd[0] - rem_mean[0]) # shrink ratio
            elif narr_hd[0] > self.nr_boundary and narr_hd[1] < narr_hd[0]:
                if narr_hd[1] > 0:
                    sr = self._shrink_ratio(narr_hd, rem_mean)
                else: 
                    sr = (self.nr_boundary - rem_mean[0])/(narr_hd[0] - rem_mean[0]) # shrink ratio
            else:
                sr = 1
            w[i] = w[i] * sr

        # confine the length of principal axes
        prn_ax = v@np.diag(w/2)  # half w because it was doubled in the previous process
        prn_ax_len = np.sqrt(np.diag(prn_ax.T@prn_ax)) # lengths of principal axes
        for i in range(3):
            if prn_ax_len[i] > self.max_rem_ax:
                sr = self.max_rem_ax / prn_ax_len[i]
            else:
                sr = 1
            w[i] = w[i] * sr
        
        cov_updated = v@np.diag((w/2)**2)@v.T

        return cov_updated


    def _confine_Wake_in_boundary(self, wake_mean, wake_cov):
        """ By definition, Wake cluster is not likely to cross the diagonal line.
        This function focuses on the ellipsoid that represents the 95% confidence area
        of Wake cluster. If this function finds any principal axis of the ellipsoid
        penetrating the diagonal line (i.e. the end-point of the principal
        axis is in the area y < x), it shrinks the length of the axis to the point on
        the constraints.
        """

        w, v = linalg.eigh(wake_cov)
        # all eigenvalues must be positive
        if np.any(w <= 0):
            raise ValueError('Invalid_Wake_Cluster')

        w = 2 * np.sqrt(w)  # 95% confidence (2SD) area
        prn_ax = v@np.diag(w)  # 3x3 matrix: each column is the principal axis
        # confine above diagonal line
        for i in range(3):
            arr_hd = wake_mean + prn_ax[:, i]  # the arrow head from the mean
            # the negative arrow head from the mean
            narr_hd = wake_mean - prn_ax[:, i]
            if arr_hd[1] < arr_hd[0]:
                sr = self._shrink_ratio(arr_hd, wake_mean)
            elif narr_hd[1] < narr_hd[0]:
                sr = self._shrink_ratio(narr_hd, wake_mean)
            else:
                sr = 1
            w[i] = w[i] * sr


        cov_updated = v@np.diag((w/2)**2)@v.T

        return cov_updated

    
    def _confine_NREM_in_boundary(self, nrem_mean, nrem_cov):
        """ By definition, NREM cluster is not likely to cross the diagonal line.
        This function focuses on the ellipsoid that represents the 95% confidence area
        of NREM cluster. If this function finds any principal axis of the ellipsoid
        penetrating the diagonal line (i.e. the end-point of the principal
        axis is in the area y > x), it shrinks the length of the axis to the point on
        the constraints.
        """

        w, v = linalg.eigh(nrem_cov)
        # all eigenvalues must be positive
        if np.any(w <= 0):
            raise ValueError('Invalid_NREM_Cluster')

        w = 2 * np.sqrt(w)  # 95% confidence (2SD) area
        prn_ax = v@np.diag(w)  # 3x3 matrix: each column is the principal axis

        # confine below diagonal line
        for i in range(3):
            arr_hd = nrem_mean + prn_ax[:, i]  # the arrow head from the mean
            # the negative arrow head from the mean
            narr_hd = nrem_mean - prn_ax[:, i]
            if arr_hd[1] > arr_hd[0]:
                sr = self._shrink_ratio(arr_hd, nrem_mean)
            elif narr_hd[1] > narr_hd[0]:
                sr = self._shrink_ratio(narr_hd, nrem_mean)
            else:
                sr = 1
            w[i] = w[i] * sr


        cov_updated = v@np.diag((w/2)**2)@v.T

        return cov_updated
 

    def _shrink_ratio(self, arr_hd, mn):
        r = (arr_hd[1] - mn[1])/(arr_hd[0]-mn[0])
        x_on_diag = (mn[1] - r*mn[0])/(1-r)
        sr = (x_on_diag - mn[0])/(arr_hd[0] - mn[0])
        return sr


    #pylint: disable = redefined-outer-name
    def _do_mstep(self, stats):
        # pylint: disable = attribute-defined-outside-init
        ghmm_stats = stats
        # pylint: disable = protected-access
        base._BaseHMM._do_mstep(self, ghmm_stats)
        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = ghmm_stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + ghmm_stats['obs'])
                           / (means_weight + denom))
 
        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2
                          + ghmm_stats['obs**2']
                          - 2 * self.means_ * ghmm_stats['obs']
                          + self.means_**2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                   self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(ghmm_stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + ghmm_stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * ghmm_stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + ghmm_stats['post'].sum()))
                elif self.covariance_type == 'full':
                    covars = ((covars_prior + cv_num) /
                              (cvweight + ghmm_stats['post'][:, None, None]))
                    confined_rem_cov = self._confine_REM_in_boundary(
                        self.means_[1], covars[1])
                    confined_wake_cov = self._confine_Wake_in_boundary(
                        self.means_[0], covars[0])
                    confined_nrem_cov = self._confine_NREM_in_boundary(
                        self.means_[2], covars[2])
                    self._covars_ = np.array(
                        [confined_wake_cov, confined_rem_cov, confined_nrem_cov])


def initialize_logger(log_file):
    logger = getLogger(FASTER2_NAME)
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
    if 'log' in globals():
        log.debug(msg)
    else:
        print(msg)


def print_log_exception(msg):
    if 'log' in globals():
        log.exception(msg)
    else:
        print(msg)


def read_mouse_info(data_dir, mouse_info_ext=None):
    """This function reads the mouse.info.csv file
    and returns a DataFrame with fixed column names.

    Args:
        data_dir (str): A path to the data directory that contains the mouse.info.csv
        mouse_info_ext (str): A sub-extention of the mouse.info.[HERE].csv


        The data directory should include two information files:
        1. exp.info.csv,
        2. mouse.info.csv,
        and one directory named "raw" that contains all EEG/EMG data to be processed

    Returns:
        DataFrame: A dataframe with a fixed column names
    """

    if mouse_info_ext:
        sub_ext = f'{mouse_info_ext}.'
    else:
        sub_ext = ''
    filepath = os.path.join(data_dir, f"mouse.info.{sub_ext}csv")

    try:
        codename = et.encode_lookup(filepath)
    except LookupError as lkup_err:
        print_log(lkup_err)
        exit(1)

    csv_df = pd.read_csv(filepath,
                         engine="python",
                         dtype={'Device label': str, 'Mouse group': str,
                                'Mouse ID': str, 'DOB': str, 'Stats report': str, 'Note': str},
                         names=["Device label", "Mouse group",
                                "Mouse ID", "DOB", "Stats report", "Note"],
                         skiprows=1,
                         header=None,
                         skipinitialspace=True,
                         encoding=codename)

    return csv_df


def read_exp_info(data_dir):
    """This function reads the exp.info.csv file
    and returns a DataFrame with fixed column names.

    Args:
        data_dir (str): A path to the data directory that contains the exp.info.csv

    Returns:
        DataFrame: A DataFrame with a fixed column names
    """

    filepath = os.path.join(data_dir, "exp.info.csv")

    csv_df = pd.read_csv(filepath,
                         engine="python",
                         dtype={"Experiment label":str, "Rack label":str,
                                "Start datetime":str, "End datetime":str, "Sampling freq":int},
                         names=["Experiment label", "Rack label", "Start datetime", 
                                "End datetime", "Sampling freq"],
                         skiprows=1,
                         header=None)

    return csv_df


def find_edf_files(data_dir):
    """returns list of edf files in the directory

    Args:
        data_dir (str): A path to the data directory

    Returns:
        [list]: A list returned by glob()
    """
    return glob(os.path.join(data_dir, '*.edf'))


def read_voltage_matrices(data_dir, device_id, sample_freq, epoch_len_sec, epoch_num,
                          start_datetime=None):
    """ This function reads data files of EEG and EMG, then returns matrices
    in the shape of (epochs, signals).

    Args:
        data_dir (str): a path to the dirctory that contains either dsi.txt/, pkl/ directory,
        or an EDF file.
        device_id (str): a transmitter ID (e.g., ID47476) or channel ID (e.g., 09).
        sample_freq (int): sampling frequency
        epoch_len_sec (int): the length of an epoch in seconds
        epoch_num (int): the number of epochs to be read.
        start_datetime (datetime): start datetime of the analysis (used only for EDF file and
        dsi.txt).

    Returns:
        (np.array(2), np.arrray(2), bool): a pair of voltage 2D matrices in a tuple
        and a switch to tell if there was pickled data.

    Note:
        This function looks into the data_dir/ and first try to read pkl files. If pkl files
        are not found, it tries to read an EDF file. If the EDF file is also not found, it
        tries to read dsi.txt files.
    """

    if os.path.exists(os.path.join(data_dir, 'pkl', f'{device_id}_EEG.pkl')):
        # if it exists, read the pkl file
        not_yet_pickled = False
        # Try to read pickled data
        pkl_path = os.path.join(data_dir, 'pkl', f'{device_id}_EEG.pkl')
        with open(pkl_path, 'rb') as pkl:
            print_log(f'Reading {pkl_path}')
            eeg_vm = pickle.load(pkl)

        pkl_path = os.path.join(data_dir, 'pkl', f'{device_id}_EMG.pkl')
        with open(pkl_path, 'rb') as pkl:
            print_log(f'Reading {pkl_path}')
            emg_vm = pickle.load(pkl)

    elif len(find_edf_files(data_dir)) > 0:
        # try to read EDF file
        not_yet_pickled = True
        # read EDF file
        edf_file = find_edf_files(data_dir)
        if len(edf_file) != 1:
            raise FileNotFoundError(
                f'Too many EDF files were found:{edf_file}. '
                'FASTER2 assumes there is only one file.')
        edf_file = edf_file[0]

        raw = mne.io.read_raw_edf(edf_file)
        measurement_start_datetime = datetime.utcfromtimestamp(
            raw.info['meas_date'][0]) + timedelta(microseconds=raw.info['meas_date'][1])
        try:
            if isinstance(start_datetime, datetime) and (measurement_start_datetime < start_datetime):
                start_offset_sec = (
                    start_datetime - measurement_start_datetime).total_seconds()
                end_offset_sec = start_offset_sec + epoch_num * epoch_len_sec
                bidx = (raw.times >= start_offset_sec) & (
                    raw.times < end_offset_sec)
                start_slice = np.where(bidx)[0][0]
                end_slice = np.where(bidx)[0][-1]+1
                eeg = raw.get_data(f'EEG{device_id}',
                                   start_slice, end_slice)[0]
                emg = raw.get_data(f'EMG{device_id}',
                                   start_slice, end_slice)[0]
            else:
                eeg = raw.get_data(f'EEG{device_id}')[0]
                emg = raw.get_data(f'EMG{device_id}')[0]
        except ValueError:
            print_log(f'Failed to extract the data of "{device_id}" from {edf_file}. '
                      f'Check the channel name: "EEG/EMG{device_id}" is in the EDF file.')
            raise
        except IndexError:
            print_log(f'Failed to extract the data of "{device_id}" from {edf_file}. '
                      f'Check the exp.info start datetime: "{start_datetime}" is consistent with EDF file. '
                      f'The start datetime of the EDF file is {measurement_start_datetime}')
            raise
        raw.close()
        try:
            eeg_vm = eeg.reshape(-1, epoch_len_sec * sample_freq)
            emg_vm = emg.reshape(-1, epoch_len_sec * sample_freq)
        except ValueError:
            print_log(f'Failed to extract {epoch_num} epochs from {edf_file}. '
                      'Check the validity of the epoch number, start datetime, '
                      'and sampling frequency.')
            raise
    elif os.path.exists(os.path.join(data_dir, 'dsi.txt')):
        # try to read dsi.txt
        not_yet_pickled = True
        try:
            dsi_reader_eeg = et.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'),
                                               f'{device_id}', 'EEG',
                                               sample_freq=sample_freq)
            dsi_reader_emg = et.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'),
                                               f'{device_id}', 'EMG',
                                               sample_freq=sample_freq)
            if isinstance(start_datetime, datetime):
                end_datetime = start_datetime + \
                    timedelta(seconds=epoch_len_sec*epoch_num)
                eeg_df = dsi_reader_eeg.read_epochs_by_datetime(
                    start_datetime, end_datetime)
                emg_df = dsi_reader_emg.read_epochs_by_datetime(
                    start_datetime, end_datetime)
            else:
                eeg_df = dsi_reader_eeg.read_epochs(1, epoch_num)
                emg_df = dsi_reader_emg.read_epochs(1, epoch_num)
            eeg_vm = eeg_df.value.values.reshape(-1,
                                                 epoch_len_sec * sample_freq)
            emg_vm = emg_df.value.values.reshape(-1,
                                                 epoch_len_sec * sample_freq)
        except FileNotFoundError:
            print_log(
                f'The dsi.txt file for {device_id} was not found in {data_dir}.')
            raise
    else:
        raise FileNotFoundError(
            f'Data file was not found for device {device_id} in {data_dir}.')

    expected_shape = (epoch_num, sample_freq * epoch_len_sec)
    if (eeg_vm.shape != expected_shape) or (emg_vm.shape != expected_shape):
        raise ValueError(f'Unexpected shape of matrices EEG:{eeg_vm.shape} or EMG:{emg_vm.shape}. '
                         f'Expected shape is {expected_shape}. Check the validity of '
                         'the data files or configurations '
                         'such as the epoch number and the sampling frequency.')

    return (eeg_vm, emg_vm, not_yet_pickled)


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

    return datetime_obj


def interpret_exp_info(exp_info_df, epoch_len_sec):
    try:
        start_datetime_str = exp_info_df['Start datetime'].values[0]
        end_datetime_str = exp_info_df['End datetime'].values[0]
        sample_freq = int(exp_info_df['Sampling freq'].values[0])
        exp_label = exp_info_df['Experiment label'].values[0]
        rack_label = exp_info_df['Rack label'].values[0]
    except KeyError as key_err:
        print_log(
            f'Failed to parse the column: {key_err} in exp.info.csv. Check the headers.')
        exit(1)

    start_datetime = interpret_datetimestr(start_datetime_str)
    end_datetime = interpret_datetimestr(end_datetime_str)

    epoch_num = int(
        (end_datetime - start_datetime).total_seconds() / epoch_len_sec)

    return (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime)


def psd(y, n_fft, sample_freq):
    return signal.welch(y, nperseg=n_fft, fs=sample_freq)[1][0:129]


def plot_hist_on_separation_axis(path2figures, d, means, covars, weights, draw_pdf_plot=False):

    if means[0] > means[1]:
        # reverse the order
        means = means[::-1]
        covars = covars[::-1]
        weights = weights[::-1]

    d_axis = np.arange(-20, 20)

    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH,
                          SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 0.1)

    ax.hist(d, bins=100, density=True)
    ax.plot(d_axis, weights[0]*stats.norm.pdf(d_axis,
                                              means[0], np.sqrt(covars[0])).ravel(), c=COLOR_WAKE)
    ax.plot(d_axis, weights[1]*stats.norm.pdf(d_axis,
                                              means[1], np.sqrt(covars[1])).ravel(), c=COLOR_NREM)
    ax.axvline(x=means[0], color='black', dashes=[2, 2])
    ax.axvline(x=means[1], color='black', dashes=[2, 2])
    ax.axvline(x=np.mean(means), color='black')

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('', fontsize=10)

    _savefig(path2figures, 'histogram_on_separation_axis', fig, draw_pdf_plot)

    return fig


def plot_scatter2D(points_2D, classes, means, covariances, colors, xlabel, ylabel, diag_line=False):
    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH,
                          SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    for i, color in enumerate(colors):

        ax.scatter(points_2D[classes == i, 0],
                   points_2D[classes == i, 1], .01, color=color)

        # Plot an ellipse to show the Gaussian component
        if (len(means) > i and len(covariances) > i):
            mean = means[i]
            covar = covariances[i]
            w, v = linalg.eigh(covar)
            w = 4. * np.sqrt(w)  # 95% confidence (2SD) area (2*radius)
            angle = np.arctan(v[1, 0] / v[0, 0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(
                mean, w[0], w[1], 180. + angle, facecolor='none', edgecolor=color)
            ax.add_patch(ell)
    if diag_line == True:
        ax.plot([-20, 20], [-20, 20], color='gray', linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    return fig


def pickle_voltage_matrices(eeg_vm, emg_vm, data_dir, device_id):
    """ To save time for reading CSV files, pickle the voltage matrices
    in pickle files.
    
    Args:
        eeg_vm (np.array): voltage matrix for EEG data
        emg_vm (np.array): voltage matrix for EMG data
        data_dir (str):  path to the directory of pickled data (pkl/)
        device_id (str): a string to identify the recording device (e.g. ID47467)
"""
    pickle_dir = os.path.join(data_dir, 'pkl/')
    os.makedirs(pickle_dir, exist_ok=True)

    # save EEG
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EEG.pkl')
    if os.path.exists(pkl_path):
        print_log(f'File already exists. Nothing to be done. {pkl_path}')
    else:
        with open(pkl_path, 'wb') as pkl:
            print_log(f'Saving the voltage matrix into {pkl_path}')
            pickle.dump(eeg_vm, pkl)

    # save EMG
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EMG.pkl')
    if os.path.exists(pkl_path):
        print_log(f'File already exists. Nothing to be done. {pkl_path} ')
    else:
        with open(pkl_path, 'wb') as pkl:
            print_log(f'Saving the voltage matrix into {pkl_path}')
            pickle.dump(emg_vm, pkl)


def pickle_powerspec_matrices(spec_norm_eeg, spec_norm_emg, result_dir_path, device_id):
    """ pickles the power spectrum density matrices for subsequent analyses

    Args:
        spec_norm_eeg (dict): a dict returned by spectrum_normalize() for EEG data
        spec_norm_emg (dict): a dict returned by spectrum_normalize() for EMG data
        bidx_unknown (np.array): an array of the boolean index
        result_dir_path (str):  path to the directory of the pickled data (PSD/)
        device_id (str): a string to identify the recording device (e.g. ID47467)
    """
    pickle_dir = os.path.join(result_dir_path, 'PSD/')
    os.makedirs(pickle_dir, exist_ok=True)

    print_log(f'Saving PSD files')

    # save EEG PSD
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EEG_PSD.pkl')
    with open(pkl_path, 'wb') as pkl:
        print_log(f'Saving the EEG PSD matrix into {pkl_path}')
        pickle.dump(spec_norm_eeg, pkl)

    # save EMG PSD
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EMG_PSD.pkl')
    with open(pkl_path, 'wb') as pkl:
        print_log(f'Saving the EMG PSD matrix into {pkl_path}')
        pickle.dump(spec_norm_emg, pkl)


def pickle_cluster_params(means2, covars2, c_means, c_covars, result_dir_path, device_id):
    """ pickles the cluster parameters
    
    Args:
        means2 (np.array(2,2)): a mean matrix of 2 stage-clusters
        covars2 (np.array(2,2,2)): a covariance matrix of 2 stage-clusters
        c_means (np.array(3,3)):  a mean matrix of 3 stage-clusters
        c_covars (np.array(3,3,3)): a covariance matrix of 3 stage-clusters
    """
    pickle_dir = os.path.join(result_dir_path, 'cluster_params/')
    os.makedirs(pickle_dir, exist_ok=True)

    # save
    pkl_path = os.path.join(pickle_dir, f'{device_id}_cluster_params.pkl')
    print_log(f'Saving the cluster parameters into {pkl_path}')
    with open(pkl_path, 'wb') as pkl:
        pickle.dump({'2stage-means': means2, '2stage-covars': covars2,
                     '3stage-means': c_means, '3stage-covars': c_covars}, pkl)

def remove_extreme_voltage(y, sample_freq):
    """An optional function to remove periodic-spike noises such as
    heart beat in EMG. Since the spikes are ofhen above 1.64 SD (upper 10%) 
    within the data region of interest, FASTER2 tries to replace those points with
    randam values.
    Note: This function is destructive i.e. it changes values of the
    given vector.


    Args:
        y (np.array(1)): a vector of voltages in an epoch
        sample_freq (int): the sampling frequency
    """
    vm = y.reshape(-1, sample_freq*2) # 2 sec
    for v in vm:
        m = np.mean(v)
        s = np.std(v)
        bidx = (abs(v) - m) > 1.64*s
        v[bidx] =  np.random.normal(m, s, np.sum(bidx))


def remove_extreme_power(y):
    """In FASTER2, the spectrum powers are normalized so that the mean and
    SD of each frequency power over all epochs become 0 and 1, respectively.
    This function removes extremely high or low powers in the normalized
    spectrum by replacing the value with a random number sampled from the
    noraml distribution: N(0, 1).

    Args:
        y (np.array(1)): a vector of normalized power spectrum

    Returns:
        float: The ratio of the replaced exreme values in the given vector.
    """
    n_len = len(y)

    bidx = (np.abs(y) > 3)  # extreme means over 3SD
    n_extr = np.sum(bidx)

    y[bidx] = np.random.normal(0, 1, n_extr)

    return n_extr / n_len


def spectrum_normalize(voltage_matrix, n_fft, sample_freq):
    """ Normalizes the power along the time axis

    Args:
        voltage_matrix (2D np.array): voltages of each epoch
        n_fft (int): a parameter of psd()
        sample_freq (int): a parameter of psd()

    Returns:
        dict: normalized PSDs and associated factors.
    """
    # power-spectrum normalization of EEG
    psd_mat = np.apply_along_axis(lambda y: psd(
        y, n_fft, sample_freq), 1, voltage_matrix)

    # If there is any zero power, replace it with a smaller value
    bidx_zero_power = (psd_mat<=0)
    if bidx_zero_power.any():
        max_power = np.max(psd_mat[~bidx_zero_power])
        peri_min_power = np.min(psd_mat[~bidx_zero_power])
        alt_min = peri_min_power/(1000*(max_power - peri_min_power)) # one-thousandth of the peri_min
        psd_mat[bidx_zero_power] = alt_min

    psd_mat = 10*np.log10(psd_mat)  # decibel-like
    psd_mean = np.apply_along_axis(np.nanmean, 0, psd_mat)
    psd_sd = np.apply_along_axis(np.nanstd, 0, psd_mat)
    spec_norm_fac = 1/psd_sd
    psd_norm_mat = np.apply_along_axis(lambda y: spec_norm_fac*(y - psd_mean),
                                       1,
                                       psd_mat)
    return {'psd': psd_norm_mat, 'mean': psd_mean, 'norm_fac': spec_norm_fac}


def projection_on_sep_axix(stage_coord_2D, sep_vec_type):
    """ projects stage_coord_2D onto the separation axis to roughly estimate
    the sleep/wake clusters. There are two types of separation axes. One is the
    "normal" axis that gives equal weight to both the high- and low-frequency powers.
    The other is the "low" axis that emphasizes more on the low-frequency power.

    Args:
        stage_coord_2D (np.array): 2D array of the stage coordination
        sep_vec_type (str): "normal" or "low".

    Raises:
        ValueError: There is only 'normal' or 'low' axis type.

    Returns:
        sep_vec: The separation vector.
        gmm: gaussian mixture model.
        d: 1D data points projected onto the separation axis.
    """
    print_log(f'Trying the \"{sep_vec_type}\" separation axis.')
    if sep_vec_type == 'normal':
        # Both the low and high freq are equally important
        sep_vec = np.array([1, -1])
        sep_vec = sep_vec.T/np.sqrt(np.dot(sep_vec, sep_vec))
    elif sep_vec_type == 'low':
        # The low freq is more important
        sep_vec = np.array([1, -0.5])
        sep_vec = sep_vec.T/np.sqrt(np.dot(sep_vec, sep_vec))
    else:
        raise ValueError('Unknown separation type specified')

    d = stage_coord_2D@sep_vec  # project onto the separation axis
    d = d.reshape(-1, 1)
    # Two states: active and quiet
    gmm = mixture.BayesianGaussianMixture(n_components=2, n_init=10)
    gmm.fit(d)

    return (sep_vec, gmm, d)


def cancel_weight_bias(stage_coord_2D):
    """ Correct the weight bias of wake/sleep clusters. FASTER2 assumes that the wake/sleep clusters
    are distributed on both sides of the diagonal line on the 2D plane (low- vs. high-frequency power
    plane). However, either cluster may lie closer to the diagonal line when it has more weight than
    the other. This is because the center of gravity of all datapoints is always closer to the larger
    cluster, and the COG approaches the diagonal line. To correct the bias, this function tries to
    estimate the midpoint between the wake/sleep clusters, and shifts all datapoints to make the estimated
    midpoint align with the diagonal line.

    Args:
        stage_coord_2D (np.array): 2D array of the stage coordination

    Returns:
        dict: A dict of estimated cluster parameters and the bias cancelled stage_coord_2D
    """

    print_log('Estimate the bias of the two cluster means')

    # Try the normal separation axis first
    s, gmm, d = projection_on_sep_axix(stage_coord_2D, 'normal')
    if gmm.converged_ == True:
        weights = gmm.weights_
        means = gmm.means_.flatten()
        covars = gmm.covariances_.flatten()
        sep_vec = s
    else:
        # backup results
        weights_bak = gmm.weights_
        means_bak = gmm.means_.flatten()
        covars_bak = gmm.covariances_.flatten()
        sep_vec_bak = s
        d_bak = d

        # when if it doesn't converge, try 'low' power focused separation axis
        s, gmm, d = projection_on_sep_axix(stage_coord_2D, 'low')
        if gmm.converged_ == True:
            weights = gmm.weights_
            means = gmm.means_.flatten()
            covars = gmm.covariances_.flatten()
            sep_vec = s
        else:
            print_log(f'Falling back to the normal seperation axis.')
            # falls back to the normal when the 'low' axis also fails
            weights = weights_bak
            means = means_bak
            covars = covars_bak
            sep_vec = sep_vec_bak
            d = d_bak

    # this is supposed to be zero if weights are completely balanced
    bias = np.mean(means[0:2])
    eb = bias * sep_vec  # x,y-axis components
    print_log(f'Estimated bias: {eb}')

    stage_coord_2D = stage_coord_2D - eb

    return {'proj_data': d, 'means': means, 'covars': covars, 'weights': weights, 'stage_coord_2D': stage_coord_2D}


def classify_active_and_NREM(stage_coord_2D):
    # Initialize active/stative(NREM) clusters by Gaussian mixture model ignoring transition probablity
    print_log('Initialize active/NREM clusters with the diagonal line')

    # projection onto the separation "line" (which is perpendicular to the separation "axis")
    stage_coord_1DD = stage_coord_2D@np.array([1, 1]).T
    bidx_over_outliers = stage_coord_1DD > (
        np.mean(stage_coord_1DD) + 3*np.std(stage_coord_1DD))
    bidx_under_outliers = stage_coord_1DD < (
        np.mean(stage_coord_1DD) - 3*np.std(stage_coord_1DD))
    bidx_valid = ~(bidx_over_outliers | bidx_under_outliers)

    # To estimate clusters, use only epochs projected within the reasonable region on the separation line (<3SD)
    def _geo_classifier(coord):
        # geometrical classifier (simple separation by the diagonal line)
        if coord[0] - coord[1] > 0:
            return 1
        else:
            return 0

    # Means and covariances of the active and NREM clusters
    geo_pred = np.array([_geo_classifier(c) for c in stage_coord_2D])
    mm_2D = np.array([
        np.mean(stage_coord_2D[geo_pred == 0 & bidx_valid], axis=0),
        np.mean(stage_coord_2D[geo_pred == 1 & bidx_valid], axis=0),
    ])
    cc_2D = np.array([
        np.cov(stage_coord_2D[geo_pred == 0 & bidx_valid], rowvar=False),
        np.cov(stage_coord_2D[geo_pred == 1 & bidx_valid], rowvar=False)
    ])

    likelihood = np.stack([multivariate_normal.pdf(stage_coord_2D, mean=mm_2D[i], cov=cc_2D[i])
                           for i in [0, 1]])
    geo_pred_proba = (likelihood / likelihood.sum(axis=0))

    return (geo_pred, geo_pred_proba.T, mm_2D, cc_2D)


def classify_active_and_NREM_by_GHMM(stage_coord_2D, pred_2D, mm_2D, cc_2D):
    # Initialize active/stative(NREM) clusters by Gaussian mixture model
    print_log('Classify active/NREM clusters with GHMM')

    weights = np.array(
        [np.sum(pred_2D == 0), np.sum(pred_2D == 1)])/len(pred_2D)

    ghmm_2D = hmm.GaussianHMM(
        n_components=2, covariance_type='full', init_params='t', params='tmcs')
    ghmm_2D.startprob_ = weights
    ghmm_2D.means_ = mm_2D
    ghmm_2D.covars_ = cc_2D

    ghmm_2D.fit(stage_coord_2D)
    ghmm_2D_pred = ghmm_2D.predict(stage_coord_2D)
    ghmm_2D_proba = ghmm_2D.predict_proba(stage_coord_2D)
    return (ghmm_2D_pred, ghmm_2D_proba, ghmm_2D.means_, ghmm_2D.covars_)


def classify_Wake_and_REM(stage_coord_active, rem_floor):
    # Classify REM and Wake in the active cluster in the 3D space  (Low freq. x High freq. x REM metric)
    print_log('Classify REM and Wake clusters with GMM')

    # exclude intermediate points between REM and Wake, and points having substantial sleep_freq power
    bidx_wake_rem = ((stage_coord_active[:, 2] > rem_floor) | (
        stage_coord_active[:, 2] < 0)) & (stage_coord_active[:, 0] < 0)
    stage_coord_wake_rem = stage_coord_active[bidx_wake_rem, :]

    # gmm for wake & REM
    gmm_wr = mixture.GaussianMixture(n_components=3, n_init=100, means_init=[
                                     [-5, 5, -10], [0, 0, 20], [0, 0, 0]])  # Wake, REM, intermediate
    gmm_wr.fit(stage_coord_wake_rem)
    ww_wr = gmm_wr.weights_
    mm_wr = gmm_wr.means_
    cc_wr = gmm_wr.covariances_
    pred_wr = gmm_wr.predict(stage_coord_active)
    pred_wr_proba = gmm_wr.predict_proba(stage_coord_active)

    # Treat the intermediate as wake
    pred_wr[pred_wr == 2] = 0
    pred_wr_proba = np.array([[x[0]+x[2], x[1]] for x in pred_wr_proba])

    # The subsequent process uses the Wake and REM clusters
    ww_wr = ww_wr[np.r_[0, 1]]
    mm_wr = mm_wr[np.r_[0, 1]]
    cc_wr = cc_wr[np.r_[0, 1]]

    if mm_wr[0, 2] > mm_wr[1, 2]:
        # flip the order of clusters to assure the order of indices 0:Wake, 1:REM
        mm_wr = np.array([mm_wr[1], mm_wr[0]])
        cc_wr = np.array([cc_wr[1], cc_wr[0]])
        pred_wr = np.array([0 if x == 1 else 1 for x in pred_wr])
        pred_wr_proba = pred_wr_proba[:, np.r_[1, 0]]

    return (pred_wr, pred_wr_proba, mm_wr, cc_wr, ww_wr)


def classify_three_stages(stage_coord, mm_3D, cc_3D, weights_3c, max_rem_prn_len):
    # pylint: disable = attribute-defined-outside-init
    # classify REM, Wake, and NREM by Gaussian HMM in the 3D space
    print_log('Classify REM, Wake, and NREM by Gaussian HMM')
    ghmm_3D = CustomedGHMM(
        n_components=3, covariance_type='full', init_params='t', params='ct')
    ghmm_3D.startprob_ = weights_3c
    ghmm_3D.means_ = mm_3D
    ghmm_3D.covars_ = cc_3D
    ghmm_3D.set_wr_boundary(0)
    ghmm_3D.set_nr_boundary(0)
    ghmm_3D.set_max_rem_ax(max_rem_prn_len)

    ghmm_3D.fit(stage_coord)
    pred_3D = ghmm_3D.predict(stage_coord)
    pred_3D_proba = ghmm_3D.predict_proba(stage_coord)
    pred_3D_mm = ghmm_3D.means_
    pred_3D_cc = ghmm_3D.covars_

    return (pred_3D, pred_3D_proba, pred_3D_mm, pred_3D_cc)


def classify_two_stages(stage_coord, pred_2D_org, mm_2D_org, cc_2D_org, mm_active, cc_active):
    ndata = len(stage_coord)
    bidx_active = (pred_2D_org == 0)
    # perform GMM to refine active/NREM classification
    pred_2D, pred_2D_proba, mm_2D, cc_2D = classify_active_and_NREM_by_GHMM(
        stage_coord[:, 0:2], pred_2D_org, mm_2D_org, cc_2D_org)

    # construct 3D means and covariances from mm_2D and mm_active with TINY (non-effective) REM cluster
    # This non-effective REM cluster is just for convenience of plotting, so has nothing to do with analytical process.
    mm_3D = np.vstack([mm_active[0], [0, 0, 100], np.mean(
        stage_coord[pred_2D == 1], axis=0)])  # Wake, REM, NREM
    cc_3D = np.vstack([[cc_active[0]], [np.diag([0.01, 0.01, 0.01])], [
                      np.cov(stage_coord[pred_2D == 1], rowvar=False)]])

    # change label of NREM from 1 to 2 so that REM can use label:1
    pred_3D = np.array([2 if x == 1 else 0 for x in pred_2D])
    idx_active = np.where(bidx_active)[0]
    # idx_REMlike = idx_active[bidx_REMlike]
    # pred_3D[idx_REMlike] = 1

    pred_3D_proba = np.zeros([ndata, 3])
    # probability of REM is always zero, but sometimes REM like.
    pred_3D_proba[:, np.r_[0, 2]] = pred_2D_proba

    return pred_2D, pred_2D_proba, mm_2D, cc_2D, pred_3D, pred_3D_proba, mm_3D, cc_3D


def classification_process(stage_coord, rem_floor):
    ndata = len(stage_coord)

    # 2-stage classification
    # classify active and NREM clusters on the 2D plane of (Low freq. x High freq.)
    pred_2D, pred_2D_proba, mm_2D, cc_2D = classify_active_and_NREM(
        stage_coord[:, 0:2])

    # Calculate the length of the longest principal axis of the active cluster
    w, v = linalg.eigh(cc_2D[0])
    w = np.sqrt(w)
    prn_ax = v@np.diag(w) # each column is the principal axis
    prn_ax_len = np.sqrt(np.diag(prn_ax.T@prn_ax))
    max_prn_ax_len = np.max(prn_ax_len)

    # Classify REM and Wake in the active cluster in the 3D space  (Low freq. x High freq. x REM metric)
    bidx_active = (pred_2D == 0)
    stage_coord_active = stage_coord[bidx_active, :]
    # pylint: disable=unused-variable
    pred_active, pred_active_proba, mm_active, cc_active, ww_active = classify_Wake_and_REM(
        stage_coord_active, rem_floor)

    # If the z values of the both clusters are negative or zero, it means there is no REM cluster
    if np.all(mm_active[:, 2] <= 0):
        # process for data NOT having effective REM cluster
        print_log('No effective REM cluster was found.')

        pred_2D, pred_2D_proba, mm_2D, cc_2D, pred_3D, pred_3D_proba, mm_3D, cc_3D = classify_two_stages(
            stage_coord, pred_2D, mm_2D, cc_2D, mm_active, cc_active)

    else:
        # process for data having effective REM culster (this is the standard process)

        # construct 3D means and covariances from mm_2D and mm_active
        # Wake, REM, NREM
        mm_3D = np.vstack(
            [mm_active, np.mean(stage_coord[pred_2D == 1], axis=0)])
        cc_3D = np.vstack(
            [cc_active, [np.cov(stage_coord[pred_2D == 1], rowvar=False)]])

        # three cluster weights; Wake, REM, NREM
        weights_3c = np.array([np.sum(pred_active == 0), np.sum(
            pred_active == 1), np.sum(pred_2D == 1)])/ndata

        # 3-stage classification: classify REM, Wake, and NREM by Gaussian HMM on the 3D space
        try:
            pred_3D, pred_3D_proba, mm_3D, cc_3D = classify_three_stages(
                stage_coord, mm_3D, cc_3D, weights_3c, max_prn_ax_len)
        except ValueError as valerr:
            if valerr.args[0] == 'Invalid_REM_Cluster':
                print_log('REM cluster is invalid.')
                pred_2D, pred_2D_proba, mm_2D, cc_2D, pred_3D, pred_3D_proba, mm_3D, cc_3D = classify_two_stages(
                    stage_coord, pred_2D, mm_2D, cc_2D, mm_active, cc_active)
            else:
                raise

    return pred_2D, pred_2D_proba, mm_2D, cc_2D, pred_3D, pred_3D_proba, mm_3D, cc_3D


def draw_scatter_plots(path2figures, stage_coord, pred2, means2, covars2, c_pred3, c_means, c_covars, draw_pdf_plot=False):
    print_log('Drawing scatter plots')

    colors = [COLOR_WAKE, COLOR_NREM]
    axes = [0, 1]
    points = stage_coord[:, np.r_[axes]]
    fig = plot_scatter2D(points, pred2, means2, covars2,
                         colors, XLABEL, YLABEL, diag_line=True)
    _savefig(path2figures, 'ScatterPlot2D_LowFreq-HighFreq_Axes_Active-NREM', fig, draw_pdf_plot)

    points_active = stage_coord[((c_pred3 == 0) | (c_pred3 == 1)), :]
    pred_active = c_pred3[((c_pred3 == 0) | (c_pred3 == 1))]

    axes = [0, 2]  # Low-freq axis & REM axis
    points_prj = stage_coord[:, np.r_[axes]]
    colors = [COLOR_WAKE, COLOR_REM, COLOR_NREM]
    mm = np.array([m[np.r_[axes]] for m in c_means[np.r_[0, 1, 2]]])
    cc = np.array([c[np.r_[axes]][:, np.r_[axes]]
                   for c in c_covars[np.r_[0, 1, 2]]])
    fig = plot_scatter2D(points_prj, c_pred3, mm,
                         cc, colors, XLABEL, ZLABEL)
    _savefig(path2figures, 'ScatterPlot2D_LowFreq-REM_axes', fig, draw_pdf_plot)

    axes = [1, 2]  # High-freq axis & REM axis
    points_prj = stage_coord[:, np.r_[axes]]
    mm = np.array([m[np.r_[axes]] for m in c_means[np.r_[0, 1, 2]]])
    cc = np.array([c[np.r_[axes]][:, np.r_[axes]]
                   for c in c_covars[np.r_[0, 1, 2]]])
    fig = plot_scatter2D(points_prj, c_pred3, mm,
                         cc, colors, YLABEL, ZLABEL)
    _savefig(path2figures, 'ScatterPlot2D_HighFreq-REM_axes', fig, draw_pdf_plot)

    axes = [0, 1]  # Low-freq axis & High-freq axis
    points_prj = stage_coord[:, np.r_[axes]]
    colors = [COLOR_WAKE, COLOR_REM, COLOR_NREM]
    mm_proj = np.array([m[np.r_[axes]] for m in c_means[np.r_[0, 1, 2]]])
    cc_proj = np.array([c[np.r_[axes]][:, np.r_[axes]]
                        for c in c_covars[np.r_[0, 1, 2]]])
    fig = plot_scatter2D(points_prj, c_pred3, mm_proj,
                         cc_proj, colors, XLABEL, YLABEL, diag_line=True)
    _savefig(path2figures, 'ScatterPlot2D_LowFreq-HighFreq_axes_Wake_REM_NREM', fig, draw_pdf_plot)

    colors = [COLOR_WAKE, COLOR_REM, COLOR_NREM]
    colors_light = [lighten_color(c) for c in colors]
    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH,
                          SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=-135)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    ax.set_xlabel(XLABEL, fontsize=10, rotation=0)
    ax.set_ylabel(YLABEL, fontsize=10, rotation=0)
    ax.set_zlabel(ZLABEL, fontsize=10, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=8)

    for c in set(c_pred3):
        t_points = stage_coord[c_pred3 == c]
        ax.scatter3D(t_points[:, 0], t_points[:, 1], min(
            ax.get_zlim()), s=0.005, color=colors_light[c])
        ax.scatter3D(t_points[:, 0], max(ax.get_ylim()),
                     t_points[:, 2], s=0.005, color=colors_light[c])
        ax.scatter3D(max(ax.get_xlim()),
                     t_points[:, 1], t_points[:, 2], s=0.005, color=colors_light[c])

        ax.scatter3D(t_points[:, 0], t_points[:, 1],
                     t_points[:, 2], s=0.01, color=colors[c])

    _savefig(path2figures, 'ScatterPlot3D', fig, draw_pdf_plot)


def _savefig(output_dir, basefilename, fig, draw_pdf_plot):
    # PNG
    filename = f'{basefilename}.png'
    fig.savefig(os.path.join(output_dir, filename), pad_inches=0,
                bbox_inches='tight', dpi=100)
    # PDF
    if draw_pdf_plot:
        filename = f'{basefilename}.pdf'
        fig.savefig(os.path.join(output_dir, 'pdf', filename), pad_inches=0,
                    bbox_inches='tight', dpi=100)


def lighten_color(hex):
    return rgb_to_hex(tuple([int(x+(255-x)*0.5) for x in hex_to_rgb(hex)]))


def hex_to_rgb(hex_code):
    h = hex_code.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb


def rgb_to_hex(rgb_tuple):
    return '#%02x%02x%02x' % rgb_tuple


def voltage_normalize(v_mat):
    v_array = v_mat.flatten()
    v_array = v_array[~np.isnan(v_array)]
    bidx_over = v_array > (np.mean(v_array)+3*np.std(v_array))
    bidx_under = v_array < (np.mean(v_array)-3*np.std(v_array))
    bidx_valid = ~(bidx_over | bidx_under)
    v_mat_norm = (
        v_mat - np.mean(v_array[bidx_valid]))/np.std(v_array[bidx_valid])

    return v_mat_norm


def main(data_dir, result_dir, pickle_input_data, epoch_len_sec, heart_beat_filter=False, no_signal_filter=False, draw_pdf_plot=False):
    """ main """

    exp_info_df = read_exp_info(data_dir)
    (epoch_num, sample_freq, exp_label, rack_label, start_datetime,
     end_datetime) = interpret_exp_info(exp_info_df, epoch_len_sec)

    # assures frequency bins compatibe among different sampling frequencies
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)
    bidx_sleep_freq = (freq_bins < 4) | ((freq_bins > 10) &
                                         (freq_bins < 20))  # without theta, 37 bins
    bidx_active_freq = (freq_bins > 30)  # 52 bins
    bidx_theta_freq = (freq_bins >= 4) & (freq_bins < 10)  # 15 bins
    bidx_delta_freq = (freq_bins < 4)  # 11 bins
    bidx_muscle_freq = (freq_bins > 30)  # 52 bins

    n_active_freq = np.sum(bidx_active_freq)
    n_sleep_freq = np.sum(bidx_sleep_freq)
    n_theta_freq = np.sum(bidx_theta_freq)
    n_delta_freq = np.sum(bidx_delta_freq)
    n_muscle_freq = np.sum(bidx_muscle_freq)

    rem_floor = np.sum(np.sqrt([n_muscle_freq, n_theta_freq]))

    mouse_info_df = read_mouse_info(data_dir)
    for i, r in mouse_info_df.iterrows():
        device_id = r[0].strip()
        mouse_group = r[1].strip()
        mouse_id = r[2].strip()
        dob = r[3]
        note = r[4]

        print_log(f'#### {FASTER2_NAME} ###################################')
        print_log(f'#### [{i+1}] Device_id: {device_id}')
        print_log(f'Reading voltages')
        print_log(
            f'Epoch num:{epoch_num}  Epoch length:{epoch_len_sec} [s]  Sampling frequency: {sample_freq} [Hz]')
        (eeg_vm_org, emg_vm_org, not_yet_pickled) = read_voltage_matrices(
            data_dir, device_id, sample_freq, epoch_len_sec, epoch_num, start_datetime)

        if (pickle_input_data and not_yet_pickled):
            # if the command line argument has the optinal flag for pickling, pickle the voltage matrices
            pickle_voltage_matrices(
                eeg_vm_org, emg_vm_org, data_dir, device_id)

        print_log('Preprocessing and calculating PSD')

        # Put NaN to no-signal epochs of EEG
        if no_signal_filter:
            print_log('Applying the optional filter on the EEG signal')
            epoch_sd = np.apply_along_axis(np.nanstd, 1 ,eeg_vm_org)
            med_sd = np.median(epoch_sd)
            bidx_no_eeg_signal = (epoch_sd / med_sd) < 0.3 # A definition of "NO signal of EEG"
            eeg_vm_org[bidx_no_eeg_signal, :] = np.nan
            print_log(f'The number of epochs with no EEG signal: {np.sum(bidx_no_eeg_signal)}')

        # recover nans in the data if possible
        nan_ratio_eeg = np.apply_along_axis(et.patch_nan, 1, eeg_vm_org)
        nan_ratio_emg = np.apply_along_axis(et.patch_nan, 1, emg_vm_org)

        # exclude unrecoverable epochs as unknown
        bidx_unknown = np.apply_along_axis(np.any, 1, np.isnan(
            eeg_vm_org)) | np.apply_along_axis(np.any, 1, np.isnan(emg_vm_org))
        eeg_vm = eeg_vm_org[~bidx_unknown, :]
        emg_vm = emg_vm_org[~bidx_unknown, :]

        # make data comparable among different mice. Not necessary for staging,
        # but convenient for subsequnet analyses.
        eeg_vm_norm = voltage_normalize(eeg_vm)
        emg_vm_norm = voltage_normalize(emg_vm)

        # remove extreme voltages (e.g. heart beat) from EMG
        if heart_beat_filter:
            print_log('Applying the optional filter on the EMG signal')
            np.apply_along_axis(remove_extreme_voltage, 1, emg_vm_norm, sample_freq)

        # power-spectrum normalization of EEG and EMG
        spec_norm_eeg = spectrum_normalize(eeg_vm_norm, n_fft, sample_freq)
        spec_norm_emg = spectrum_normalize(emg_vm_norm, n_fft, sample_freq)
        psd_norm_mat_eeg = spec_norm_eeg['psd']
        psd_norm_mat_emg = spec_norm_emg['psd']

        # remove extreme powers
        extrp_ratio_eeg = np.apply_along_axis(
            remove_extreme_power, 1, psd_norm_mat_eeg)
        extrp_ratio_emg = np.apply_along_axis(
            remove_extreme_power, 1, psd_norm_mat_emg)

        # save the PSD matrices and associated factors for subsequent analyses
        ## set bidx_unknown; other factors were set by spectrum_normalize()
        spec_norm_eeg['bidx_unknown'] = bidx_unknown
        spec_norm_emg['bidx_unknown'] = bidx_unknown
        pickle_powerspec_matrices(
            spec_norm_eeg, spec_norm_emg, result_dir, device_id)

        # spread epochs on the 3D (Low freq. x High freq. x REM metric) space
        psd_mat = np.concatenate([
            psd_norm_mat_eeg.reshape(*psd_norm_mat_eeg.shape, 1),
            psd_norm_mat_emg.reshape(*psd_norm_mat_emg.shape, 1)
        ], axis=2)
        stage_coord = np.array([(
            np.sum(y[bidx_sleep_freq, 0])/np.sqrt(n_sleep_freq),
            np.sum(y[bidx_active_freq, 0])/np.sqrt(n_active_freq),
            np.sum(y[bidx_theta_freq, 0])/np.sqrt(n_theta_freq)-np.sum(y[bidx_delta_freq, 0]) /
            np.sqrt(n_delta_freq) -
            np.sum(y[bidx_muscle_freq, 1]) / np.sqrt(n_muscle_freq)
        ) for y in psd_mat])
        ndata = len(stage_coord)

        # cancel the weight bias of active/NREM clusters
        cwb = cancel_weight_bias(stage_coord[:, 0:2])
        stage_coord[:, 0:2] = cwb['stage_coord_2D']

        # run the classification process
        try:
            # pylint: disable=unused-variable
            pred_2D, pred_2D_proba, means_2D, covars_2D, pred_3D, pred_3D_proba, means_3D, covars_3D = classification_process(
                stage_coord, rem_floor)
        except ValueError as val_err:
            print_log('Encountered an unhandlable analytical error during the staging. Check the '
                      'date validity of the mouse.')
            print_log(traceback.format_exc())

            continue

        # output staging result
        stage_proba = np.zeros(3*epoch_num).reshape(epoch_num, 3)
        proba_REM = pred_3D_proba[:, 1]
        proba_WAKE = pred_3D_proba[:, 0]
        proba_NREM = pred_3D_proba[:, 2]
        stage_proba[~bidx_unknown, 0] = proba_REM
        stage_proba[~bidx_unknown, 1] = proba_WAKE
        stage_proba[~bidx_unknown, 2] = proba_NREM

        stage_call = np.repeat('Unknown', epoch_num)
        stage_call[~bidx_unknown] = np.array(
            [STAGE_LABELS[y] for y in pred_3D])

        # Print a brief result
        print_log(f'2-stage means:\n {means_2D}')
        print_log(f'2-stage covars:\n {covars_2D}')
        print_log('\n')
        print_log(f'3-stage means:\n{means_3D}')
        print_log(f'3-stage covars:\n{covars_3D}')

        print_log(f'[{i+1}] Device ID:{device_id}  REM:{1440*np.sum(stage_call=="REM")/ndata:.2f} '
                  f'NREM:{1440*np.sum(stage_call=="NREM")/ndata:.2f} '
                  f'Wake:{1440*np.sum(stage_call=="Wake")/ndata:.2f}')

        # Compose stage table
        extreme_power_ratio = np.zeros(2*epoch_num).reshape(epoch_num, 2)
        extreme_power_ratio[~bidx_unknown, 0] = extrp_ratio_eeg
        extreme_power_ratio[~bidx_unknown, 1] = extrp_ratio_emg

        stage_table = pd.DataFrame({'Stage': stage_call,
                                    'REM probability': stage_proba[:, 0],
                                    'NREM probability': stage_proba[:, 2],
                                    'Wake probability': stage_proba[:, 1],
                                    'NaN ratio EEG-TS': nan_ratio_eeg,
                                    'NaN ratio EMG-TS': nan_ratio_emg,
                                    'Outlier ratio EEG-TS': extreme_power_ratio[:, 0],
                                    'Outlier ratio EMG-TS': extreme_power_ratio[:, 1]})
        stage_file_path = os.path.join(
            result_dir, f'{device_id}.faster2.stage.csv')

        with open(stage_file_path, 'w', encoding='UTF-8') as f:
            f.write(f'# Exp label: {exp_label} Recorded at {rack_label}\n')
            f.write(
                f'# Device ID: {device_id} Mouse group: {mouse_group} Mouse ID: {mouse_id} DOB: {dob}\n')
            f.write(
                f'# Start: {start_datetime} End: {end_datetime} Note: {note}\n')
            f.write(f'# Epoch num: {epoch_num}  Epoch length: {epoch_len_sec} [s]\n')
            f.write(f'# Sampling frequency: {sample_freq} [Hz]\n')
            f.write(f'# Staged by {FASTER2_NAME}\n')
        with open(stage_file_path, 'a') as f:
            stage_table.to_csv(f, header=True, index=False,
                               line_terminator='\n')

        path2figures = os.path.join(result_dir, 'figure', f'{device_id}')
        os.makedirs(path2figures, exist_ok=True)
        if draw_pdf_plot:
            os.makedirs(os.path.join(path2figures, 'pdf'), exist_ok=True)

        # draw the bias histogram
        plot_hist_on_separation_axis(
            path2figures, cwb['proj_data'], cwb['means'], cwb['covars'], cwb['weights'], draw_pdf_plot) 

        # draw scatter plots
        draw_scatter_plots(path2figures, stage_coord, pred_2D,
                           means_2D, covars_2D, pred_3D, means_3D, covars_3D, draw_pdf_plot)

        # pickle cluster parameters
        pickle_cluster_params(means_2D, covars_2D, means_3D,
                              covars_3D, result_dir, device_id)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir",
                        help="path to the directory of input data", default="data")
    parser.add_argument("-r", "--result_dir",
                        help="path to the directory of staging result", default="result")
    parser.add_argument("-l", "--epoch_len_sec", help="epoch length in second", default=8)
    parser.add_argument("-p", "--pickle_input_data",
                        help="flag to pickle input data", action='store_true')
    parser.add_argument("-D", "--draw_pdf_plot", help="flag to draw PDF plots", action='store_true')
    parser.add_argument("-f", "--heart_beat_filter", help="flag to apply the heart beat filter", action='store_true')
    parser.add_argument("-n", "--no_signal_filter", help="flag to apply no signal filter", action='store_true')

    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    pickle_input_data = args.pickle_input_data
    os.makedirs(result_dir, exist_ok=True)

    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(result_dir, f'stage.{dt_str}.log'))

    print_log(f'[{dt_str} - {FASTER2_NAME} - {sys.modules[__name__].__file__}] Started in : {os.path.dirname(os.path.abspath(args.data_dir))}')

    try:
        main(args.data_dir, result_dir, args.pickle_input_data, int(args.epoch_len_sec), args.heart_beat_filter, args.no_signal_filter, args.draw_pdf_plot)
    except Exception as e:
        print_log_exception('Unhandled exception occured')

    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    print_log(f'[{dt_str} - {sys.modules[__name__].__file__}] Ended')