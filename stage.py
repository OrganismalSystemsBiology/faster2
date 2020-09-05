# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import faster2lib.eeg_tools as et
import re
from datetime import datetime, timedelta
from scipy import signal
from hmmlearn import hmm
from sklearn import mixture
import matplotlib as mpl
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from scipy import linalg, stats
from scipy.spatial import distance
import pickle
from glob import glob
import mne
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter


FASTER2_NAME = 'FASTER2'
LOG_FILE_NAME = 'stage.log'
EPOCH_LEN_SEC = 8
STAGE_LABELS = ['Wake', 'REM', 'NREM']
XLABEL = 'Total low-freq. log-powers'
YLABEL = 'Total high-freq. log-powers'
ZLABEL = 'REM metric'
SCATTER_PLOT_FIG_WIDTH = 6   # inch
SCATTER_PLOT_FIG_HEIGHT = 6  # inch
FIG_DPI = 100 # dot per inch
COLOR_WAKE = '#EF5E26'
COLOR_NREM = '#23B4EF'
COLOR_REM  = 'olivedrab' # #6B8E23
COLOR_LIGHT = 'gold' # #FFD700
COLOR_DARK = 'dimgray' # #696969
COLOR_DARKLIGHT = 'lightgray' # light hours in DD condition
DEFAULT_START_PROBA = np.array([0.1, 0.45, 0.45])
DEFAULT_MEAN_STAGE_COORDS = np.array([[-20, 10, 100], [-20, 20, -50], [20, -20, 0]])
DEFAULT_TRANSMAT = np.array([[8.73223739e-01, 6.53422888e-02, 6.14339721e-02],
                             [7.40251368e-04, 9.68070346e-01, 3.11894024e-02],
                             [1.00730294e-02, 2.49010231e-02, 9.65025948e-01]])


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

    return(logger)


def print_log(msg):
    if 'log' in globals():
        log.debug(msg)
    else:
        print(msg)


def read_mouse_info(data_dir):      
    """This function reads the mouse.info.csv file 
    and returns a DataFrame with fixed column names.

    Args:
        data_dir (str): A path to the data directory that contains the mouse.info.csv


        The data directory should include two information files:
        1. exp.info.csv,
        2. mouse.info.csv, 
        and one directory named "raw" that contains all EEG/EMG data to be processed
    
    Returns:
        DataFrame: A dataframe with a fixed column names
    """

    filepath = os.path.join(data_dir, "mouse.info.csv")

    try:
        codename = et.encode_lookup(filepath)
    except LookupError as e:
        print_log(e)
        exit(1)

    csv_df = pd.read_csv(filepath,
                         engine="python",
                         dtype={'Device label': str, 'Mouse group': str,
                                'Mouse ID': str, 'DOB': str, 'Stats report':str, 'Note': str},
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
                         names=["Experiment label", "Rack label", "Start datetime", "End datetime", "Sampling freq"], 
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


def read_voltage_matrices(data_dir, device_id, sample_freq, epoch_len_sec, epoch_num, start_datetime=None):
    """ This function reads data files of EEG and EMG, then returns matrices
    in the shape of (epochs, signals).
    
    Args:
        data_dir (str): a path to the dirctory that contains either dsi.txt/, pkl/ directory, or an EDF file.
        device_id (str): a transmitter ID (e.g., ID47476) or channel ID (e.g., 09). 
        sample_freq (int): sampling frequency 
        epoch_len_sec (int): the length of an epoch in seconds
        epoch_num (int): the number of epochs to be read.
        start_datetime (datetime): start datetime of the analysis (used only for EDF file and dsi.txt).
    
    Returns:
        (np.array(2), np.arrray(2), bool): a pair of voltage 2D matrices in a tuple
        and a switch to tell if there was pickled data.

    Note:
        This function looks into the data_dir/ and first try to read pkl files. If pkl files are not found,
        it tries to read an EDF file. If the EDF file is also not found, it tries to read dsi.txt files.
    """

    if os.path.exists(os.path.join(data_dir, 'pkl', f'{device_id}_EEG.pkl')):
        # if it exists, read the pkl file
        not_yet_pickled = False
        # Try to read pickled data
        pkl_path = os.path.join(data_dir, 'pkl', f'{device_id}_EEG.pkl')
        with open(pkl_path, 'rb') as pkl:
            print_log(f'reading {pkl_path}')
            eeg_vm = pickle.load(pkl)
        
        pkl_path = os.path.join(data_dir, 'pkl', f'{device_id}_EMG.pkl')
        with open(pkl_path, 'rb') as pkl:
            print_log(f'reading {pkl_path}')
            emg_vm = pickle.load(pkl)

    elif len(find_edf_files(data_dir))>0:
        # try to read EDF file
        not_yet_pickled = True
        # read EDF file
        edf_file = find_edf_files(data_dir)
        if len(edf_file) != 1:
            raise FileNotFoundError(f'Too many EDF files were found:{edf_file}. FASTER2 assumes there is only one file.')
        edf_file = edf_file[0]

        raw = mne.io.read_raw_edf(edf_file)
        measurement_start_datetime = datetime.utcfromtimestamp(raw.info['meas_date'][0]) + timedelta(microseconds=raw.info['meas_date'][1])
        try:
            if isinstance(start_datetime, datetime) and (measurement_start_datetime < start_datetime):
                start_offset_sec = (start_datetime - measurement_start_datetime).total_seconds()
                end_offset_sec = start_offset_sec + epoch_num * epoch_len_sec
                bidx = (raw.times >= start_offset_sec) & (raw.times < end_offset_sec)
                start_slice = np.where(bidx)[0][0]
                end_slice = np.where(bidx)[0][-1]+1
                eeg = raw.get_data(f'EEG{device_id}', start_slice, end_slice)[0]
                emg = raw.get_data(f'EMG{device_id}', start_slice, end_slice)[0]
            else:
                eeg = raw.get_data(f'EEG{device_id}')[0]
                emg = raw.get_data(f'EMG{device_id}')[0]
        except ValueError:
            print_log(f'Failed to extract the data of "{device_id}" from {edf_file}. '\
                  f'Check the channel name: "EEG/EMG{device_id}" is in the EDF file.')
            raise
        raw.close()
        try:
            eeg_vm = eeg.reshape(-1, epoch_len_sec * sample_freq)
            emg_vm = emg.reshape(-1, epoch_len_sec * sample_freq)
        except ValueError:
            print_log(f'Failed to extract {epoch_num} epochs from {edf_file}. '\
                   'Check the validity of the epoch number, start datetime, and sampling frequency.')
            raise
    elif os.path.exists(os.path.join(data_dir, 'dsi.txt')):
        # try to read dsi.txt
        not_yet_pickled = True
        try:
            dsi_reader_eeg = et.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'), 
                                            f'{device_id}', 
                                            'EEG', 
                                            sample_freq=sample_freq)
            dsi_reader_emg = et.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'), 
                                            f'{device_id}', 
                                            'EMG', 
                                            sample_freq=sample_freq)
            if isinstance(start_datetime, datetime):
                end_datetime = start_datetime + timedelta(seconds=epoch_len_sec*epoch_num)
                eeg_df = dsi_reader_eeg.read_epochs_by_datetime(start_datetime, end_datetime)
                emg_df = dsi_reader_emg.read_epochs_by_datetime(start_datetime, end_datetime)
            else:
                eeg_df = dsi_reader_eeg.read_epochs(1, epoch_num)
                emg_df = dsi_reader_emg.read_epochs(1, epoch_num)
            eeg_vm = eeg_df.value.values.reshape(-1, epoch_len_sec * sample_freq)
            emg_vm = emg_df.value.values.reshape(-1, epoch_len_sec * sample_freq)
        except FileNotFoundError:
            print_log(f'The dsi.txt file for {device_id} was not found in {data_dir}.')        
            raise
    else:
        raise FileNotFoundError(f'Data file was not found for device {device_id} in {data_dir}.')
        

    expected_shape = (epoch_num, sample_freq * epoch_len_sec)
    if (eeg_vm.shape != expected_shape) or (emg_vm.shape != expected_shape):
        raise ValueError(f'Unexpected shape of matrices EEG:{eeg_vm.shape} or EMG:{emg_vm.shape}. '\
                         f'Expected shape is {expected_shape}. Check the validity of the data files or configurations '\
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

    return(datetime_obj)
    

def interpret_exp_info(exp_info_df):
    try:
        start_datetime_str = exp_info_df['Start datetime'].values[0]
        end_datetime_str = exp_info_df['End datetime'].values[0]
        sample_freq = exp_info_df['Sampling freq'].values[0]
        exp_label = exp_info_df['Experiment label'].values[0]
        rack_label = exp_info_df['Rack label'].values[0]
    except KeyError as e:
        print_log(f'Failed to parse the column: {e} in exp.info.csv. Check the headers.')
        exit(1)

    start_datetime = interpret_datetimestr(start_datetime_str)
    end_datetime = interpret_datetimestr(end_datetime_str)

    epoch_num = int((end_datetime - start_datetime).total_seconds() / EPOCH_LEN_SEC)


    
    return (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime)


def psd(y, n_fft, sample_freq):
    return signal.welch(y, nfft=n_fft, fs=sample_freq)[1][0:129]


def plot_hist_on_separation_axis(path2figures, d, means, covars, weights):

    if means[0] > means[1]:
        # reverse the order
        means = means[::-1]
        covars = covars[::-1]
        weights = weights[::-1]

    d_axis = np.arange(-150,150)

    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH, SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_xlim(-150, 150)
    ax.set_ylim(0, 0.02)

    ax.hist(d, bins=100, density=True)
    ax.plot(d_axis, weights[0]*stats.norm.pdf(d_axis,means[0],np.sqrt(covars[0])).ravel(), c=COLOR_WAKE)
    ax.plot(d_axis, weights[1]*stats.norm.pdf(d_axis,means[1],np.sqrt(covars[1])).ravel(), c=COLOR_NREM)
    ax.axvline(x=means[0], color='black', dashes=[2,2])
    ax.axvline(x=means[1], color='black', dashes=[2,2])
    ax.axvline(x=np.mean(means), color='black')

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('', fontsize=10)

    fig.savefig(os.path.join(path2figures,'histogram_on_separation_axis.png'))

    return fig


def plot_scatter2D(points_2D, classes, means, covariances, colors, xlabel, ylabel, diag_line=False):
    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH, SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    
    for i, color in enumerate(colors):

        ax.scatter(points_2D[classes == i, 0], points_2D[classes == i, 1], .01, color=color)

        # Plot an ellipse to show the Gaussian component
        if (len(means)>i and len(covariances)>i):
            mean = means[i]
            covar = covariances[i]
            w, v = linalg.eigh(covar)
            w = 4. * np.sqrt(w) # 95% confidence (2SD) area
            angle = np.arctan(v[0, 1] / v[0, 0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, w[0], w[1], 180. + angle, facecolor='none', edgecolor=color)
            ax.add_patch(ell)
    if diag_line==True:
        ax.plot([-150, 150], [-150, 150], color='gray',linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    return fig


def shrink_rem_cluster(means, covar):
    """ By definition, REM epochs should not be z<0. This function focuses on the 
    ellipsoid that represents the 99.7% confidence area of REM cluster. If this function  
    finds any ellipsoid axis penetrating the xy plane (i.e. the end-point is 
    at z<0), it shrinks the length of the most-negative axis to point at z = 0 and 
    lengths of other axes according to their z-contributions.
    """
    z_mean = means[2]

    W, V = np.linalg.eigh(covar) # W: eigen values, V:eigen vectors (unit length)
    
    if np.any(W<=0):
        # not processable if there is zero or negative component of W
        print_log(f'Negative or zero component was found in eigen values of the covariance: {W}')
        covar_updated = np.array([]) 

    elif np.any(z_mean - np.abs(V[2,:])*3*np.sqrt(W) <0):
        idx_of_maxZ = np.argmax(np.abs(V[2,:]*np.sqrt(W))) # find the maxZ-axis: an axis with the maximum abs(Z)

        new_w = (z_mean/(3*np.abs(V[2, idx_of_maxZ])))**2 # 3SD (99.7% confidnece area) of the new maxZ-axis points at z=0

        sr = new_w / W[idx_of_maxZ] # shrink ratio
        zc = np.abs(V[2,:])*3*np.sqrt(W) / (np.abs(V[2,:])*3*np.sqrt(W))[idx_of_maxZ] # z contributions of all axes relative to the maxZ-axis
        sh_axes = 1 - (1-sr)*zc # shrink ratios of each axis

        W_updated = W * sh_axes

        covar_updated = V@np.diag(W_updated)@V.T
    else:
        # if no axis found under z=0, return a zero-size array
        covar_updated = np.array([])

    return covar_updated
 

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
        print_log(f'file already exists. Nothing to be done. {pkl_path}')
    else:
        with open(pkl_path, 'wb') as pkl:
            print_log(f'saving the voltage matrix into {pkl_path}')
            pickle.dump(eeg_vm, pkl)
    
    # save EMG
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EMG.pkl')
    if os.path.exists(pkl_path):
        print_log(f'file already exists. Nothing to be done. {pkl_path} ')
    else:
        with open(pkl_path, 'wb') as pkl:
            print_log(f'saving the voltage matrix into {pkl_path}')
            pickle.dump(emg_vm, pkl)


def pickle_powerspec_matrices(spec_norm_eeg, spec_norm_emg, bidx_unknown, result_dir, device_id):
    """ pickles the power spectrum density matrices for subsequent analyses
    
    Args:
        spec_norm_eeg (dict): a dict returned by spectrum_normalize() for EEG data
        spec_norm_emg (dict): a dict returned by spectrum_normalize() for EMG data
        bidx_unknown (np.array): an array of the boolean index
        result_dir (str):  path to the directory of the pickled data (PSD/)
        device_id (str): a string to identify the recording device (e.g. ID47467)
    """
    pickle_dir = os.path.join(result_dir, 'PSD/')
    os.makedirs(pickle_dir, exist_ok=True)

    # save EEG PSD
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EEG_PSD.pkl')
    if os.path.exists(pkl_path):
        print_log(f'file already exists. Nothing to be done. {pkl_path}')
    else:
        with open(pkl_path, 'wb') as pkl:
            print_log(f'saving the EEG PSD matrix into {pkl_path}')
            pickle.dump(spec_norm_eeg, pkl)
    
    # save EMG PSD
    pkl_path = os.path.join(pickle_dir, f'{device_id}_EMG_PSD.pkl')
    if os.path.exists(pkl_path):
        print_log(f'file already exists. Nothing to be done. {pkl_path} ')
    else:
        with open(pkl_path, 'wb') as pkl:
            print_log(f'saving the EMG PSD matrix into {pkl_path}')
            pickle.dump(spec_norm_emg, pkl)


def pickle_cluster_params(means2, covars2, c_means, c_covars, result_dir, device_id):
    """ pickles the cluster parameters
    
    Args:
        means2 (np.array(2,2)): a mean matrix of 2 stage-clusters  
        covars2 (np.array(2,2,2)): a covariance matrix of 2 stage-clusters
        c_means (np.array(3,3)):  a mean matrix of 3 stage-clusters
        c_covars (np.array(3,3,3)): a covariance matrix of 3 stage-clusters
    """
    pickle_dir = os.path.join(result_dir, 'cluster_params/')
    os.makedirs(pickle_dir, exist_ok=True)

    # save
    pkl_path = os.path.join(pickle_dir, f'{device_id}_cluster_params.pkl')
    with open(pkl_path, 'wb') as pkl:
        print_log(f'saving the cluster parameters into {pkl_path}')
        pickle.dump({'2stage-means': means2, '2stage-covars': covars2,
                     '3stage-means': c_means, '3stage-covars': c_covars}, pkl)


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

    bidx = (np.abs(y) > 3) # extreme means over 3SD
    n_extr = np.sum(bidx)

    y[bidx] = np.random.normal(0, 1, n_extr)
    
    return n_extr / n_len


def spectrum_normalize(voltage_matrix, n_fft, sample_freq):
    # power-spectrum normalization of EEG
    psd_mat = np.apply_along_axis(lambda y: psd(y, n_fft, sample_freq), 1, voltage_matrix)
    psd_mat = 10*np.log10(psd_mat) # decibel-like
    psd_mean = np.apply_along_axis(np.nanmean, 0, psd_mat)
    psd_sd = np.apply_along_axis(np.nanstd, 0, psd_mat)
    spec_norm_fac = 1/psd_sd
    psd_norm_mat = np.apply_along_axis(lambda y: spec_norm_fac*(y - psd_mean),
                                                1,
                                                psd_mat)
    return {'psd':psd_norm_mat, 'mean':psd_mean, 'norm_fac':spec_norm_fac}


def cancel_weight_bias(stage_coord_2D):
    # Estimate the center of the two clusters
    print_log('Estimate the bias of the two cluster means')
    d = stage_coord_2D@np.array([1,-1]).T # project onto the separation axis
    d = d.reshape(-1,1)
    gmm = mixture.GaussianMixture(n_components=2, n_init=10) #active, stative
    gmm.fit(d)
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covars = gmm.covariances_.flatten()

    bias = np.mean(means[0:2]) # this is supposed to be zero if weights are completely balanced
    s = bias/np.sqrt(2) # x,y-axis components
    print_log(f'Estimated bias: {s}')

    stage_coord_2D = stage_coord_2D + [-s, s]

    return {'proj_data':d, 'means':means, 'covars':covars, 'weights':weights, 'stage_coord_2D':stage_coord_2D}


def classify_active_and_NREM(stage_coord_2D):
    # Initialize active/stative(NREM) clusters by Gaussian mixture model ignoring transition probablity
    print_log('Initialize active/NREM clusters with GMM')
    gmm_2D = mixture.GaussianMixture(n_components=2, covariance_type='full', n_init=10, means_init=[[-20,20],[20,-20]])
    gmm_2D.fit(stage_coord_2D)

    gmm_2D_pred = gmm_2D.predict(stage_coord_2D)
    mm_2D = gmm_2D.means_
    cc_2D = gmm_2D.covariances_
    ww_2D = gmm_2D.weights_
    if (mm_2D[0,1] < mm_2D[1,1]) or (mm_2D[0,0] > mm_2D[1,0]):
        # flip the order of clusters if necessary
        mm_2D = np.array([mm_2D[1],mm_2D[0]])
        cc_2D = np.array([cc_2D[1],cc_2D[0]])
        ww_2D = np.array([ww_2D[1],ww_2D[0]])
        gmm_2D_pred = np.array([0 if x==1 else 1 for x in gmm_2D_pred])
    print_log(f'Means:\n{mm_2D}')
    print_log(f'Covariances:\n{cc_2D}')

    # classify active and NREM stages by Gaussian HMM on the 2D plane of (active x sleep)
    print_log('Refine active/NREM clusters with Gaussian HMM')
    ghmm_2D = hmm.GaussianHMM(n_components=2, covariance_type='full', init_params='t', params='st')
    ghmm_2D.startprob_ = ww_2D
    ghmm_2D.means_ = mm_2D
    ghmm_2D.covars_ = cc_2D
    ghmm_2D.fit(stage_coord_2D)
    print_log(f'Means:\n{ghmm_2D.means_}')
    print_log(f'Covariances:\n{ghmm_2D.covars_}')
    pred_2D = ghmm_2D.predict(stage_coord_2D)
    pred_2D_proba = ghmm_2D.predict_proba(stage_coord_2D)

    return (pred_2D, pred_2D_proba, ghmm_2D.means_, ghmm_2D.covars_)


def classify_Wake_and_REM(stage_coord_active):
    # Classify REM and Wake in the active cluster in the 3D space  (Low freq. x High freq. x REM metric)
    gmm_active = mixture.GaussianMixture(n_components=2, n_init=10, means_init=[[-20,20,-50],[0,0,100]], weights_init=[0.9, 0.1]) #active, intermdeidate, stative
    gmm_active.fit(stage_coord_active)
    ww_active = gmm_active.weights_
    mm_active = gmm_active.means_
    cc_active = gmm_active.covariances_
    pred_active = gmm_active.predict(stage_coord_active)
    pred_active_proba = gmm_active.predict_proba(stage_coord_active)

    if mm_active[0,2] > mm_active[1,2]:
        # flip the order of clusters to assure the order of indices 0:Wake, 1:REM
        mm_active = np.array([mm_active[1],mm_active[0]])
        cc_active = np.array([cc_active[1],cc_active[0]])
        pred_active = np.array([0 if x==1 else 1 for x in pred_active])
        pred_active_proba = pred_active_proba[:,np.r_[1,0]]

    return (pred_active, pred_active_proba, mm_active, cc_active, ww_active)


def find_REMlike_epochs(stage_coord_active, mm_active, cc_active):
    # Return REM-like epochs assuming a point is REM if it is above xy-planne and 
    # 3SD (Mahalanobis distance) away from wake cluster's mean
    icc = linalg.inv(cc_active[0])
    bidx_REMlike = np.array([distance.mahalanobis(mm_active[0], x ,icc)>3 and x[2]>0 for x in stage_coord_active])

    return bidx_REMlike


def classify_three_stages(stage_coord_nremflat, mm_3D, cc_3D, weights_3c):
    # classify REM, Wake, and NREM by Gaussian HMM on the 3D space
    ghmm_3D = hmm.GaussianHMM(n_components=3, covariance_type='full', init_params='t', params='st')
    ghmm_3D.startprob_ = weights_3c
    ghmm_3D.means_ = mm_3D
    ghmm_3D.covars_ = cc_3D

    ghmm_3D.fit(stage_coord_nremflat)
    pred_3D = ghmm_3D.predict(stage_coord_nremflat)
    pred_3D_proba = ghmm_3D.predict_proba(stage_coord_nremflat)

    return (pred_3D, pred_3D_proba)


def classification_process(stage_coord):
    ndata = len(stage_coord)

    # 2-stage classification
    # classify active and NREM clusters on the 2D plane of (Low freq. x High freq.)
    pred_2D, pred_2D_proba, mm_2D, cc_2D = classify_active_and_NREM(stage_coord[:, 0:2])

    # Arrange the stage_coord by compressing NREM (pred==1) epochs on xy-plane (z=0), leaving
    # only the active epochs (pred==0) expanded along the z-axis
    stage_coord_nremflat = np.array([[y[0], y[1], y[2] if pred_2D[i] == 0 else 0]
                                for i, y in enumerate(stage_coord)])    # Arrange the stage_coord by compressing NREM (pred==1) epochs on xy-plane (z=0), leaving

    # Classify REM and Wake in the active cluster in the 3D space  (Low freq. x High freq. x REM metric)
    bidx_active = (pred_2D == 0)
    stage_coord_active = stage_coord[bidx_active,:]
    pred_active, pred_active_proba, mm_active, cc_active, ww_active = classify_Wake_and_REM(stage_coord_active)

    # If the z values of the both clusters are negative or zero, it means there is no REM cluster
    if np.all(mm_active[:,2]<=0):
        # process for data NOT having effective REM cluster
        print_log('No effective REM cluster was found.')
        bidx_REMlike = find_REMlike_epochs(stage_coord_active, mm_active, cc_active)
        print_log(f'REM like epochs were found: {np.sum(bidx_REMlike)} epochs.')

        # construct 3D means and covariances from mm_2D and mm_active with TINY (non-effective) REM cluster
        mm_3D =  np.vstack([mm_active[0], [0,0,0], np.hstack([mm_2D[1], 0])]) # Wake, REM, NREM
        cc_3rdD = np.vstack([np.hstack([cc_2D[1], [[0], [0]]]),[0, 0, 0.1]]) # 0.1 is just an arbitral number
        cc_3D = np.vstack([[cc_active[0]], [np.diag([0.1, 0.1, 0.1])], [cc_3rdD]]) 

        pred_3D = np.array([2 if x==1 else 0 for x in pred_2D]) # change label of NREM from 1 to 2 so that REM can use label:1
        idx_active = np.where(bidx_active)[0]
        idx_REMlike = idx_active[bidx_REMlike]
        pred_3D[idx_REMlike] = 1

        pred_3D_proba = np.zeros([ndata, 3])
        pred_3D_proba[:, np.r_[0,2]] = pred_2D_proba # probability of REM is always zero, but sometimes REM like.

    else:
        # process for data having effective REM culster (standard)
        # try to correct REM cluster by shrinking it if the ellipsoid axis crosses the xy-plane to negative
        covar_REM_updated = shrink_rem_cluster(mm_active[1], cc_active[1])
        if covar_REM_updated.size > 0:
            cc_active[1] = covar_REM_updated

        # three cluster weights
        weights_3c = np.array([np.sum(pred_2D==1), np.sum(pred_active==0), np.sum(pred_active==1)])/ndata

        # construct 3D means and covariances from mm_2D and mm_active
        mm_3D = np.vstack([mm_active, np.hstack([mm_2D[1], 0])]) # Wake, REM, NREM
        cc_3rdD = np.vstack([np.hstack([cc_2D[1], [[0], [0]]]),[0, 0, 0.1]]) # 0.1 is just an arbitoral z-axis component
        cc_3D = np.vstack([cc_active, [cc_3rdD]])

        # 3-stage classification: classify REM, Wake, and NREM by Gaussian HMM on the 3D space
        pred_3D, pred_3D_proba = classify_three_stages(stage_coord_nremflat, mm_3D, cc_3D, weights_3c)

    return pred_2D, pred_2D_proba, mm_2D, cc_2D, stage_coord_nremflat, pred_3D, pred_3D_proba, mm_3D, cc_3D


def draw_scatter_plots(path2figures, stage_coord, pred2, means2, covars2, stage_coord_expacti, c_pred3, c_means, c_covars):
    colors =  [COLOR_WAKE, COLOR_NREM] 
    axes = [0, 1]
    points = stage_coord[:, np.r_[axes]]
    fig = plot_scatter2D(points, pred2, means2, covars2, colors, XLABEL, YLABEL, diag_line=True)
    fig.savefig(os.path.join(path2figures,'ScatterPlot2D_LowFreq-HighFreq_Axes_Active-NREM.png'))

    points_active = stage_coord_expacti[((c_pred3==0) | (c_pred3==1)), :]
    pred_active = c_pred3[((c_pred3==0) | (c_pred3==1))]

    axes = [0, 2] # Low-freq axis & REM axis
    points_prj = points_active[:, np.r_[axes]]
    colors =  [COLOR_WAKE, COLOR_REM]
    mm = np.array([m[np.r_[axes]] for m in c_means[np.r_[0,1]]])
    cc = np.array([c[np.r_[axes]][:,np.r_[axes]] for c in c_covars[np.r_[0,1]]])
    fig = plot_scatter2D(points_prj, pred_active, mm , cc, colors, XLABEL, ZLABEL)
    fig.savefig(os.path.join(path2figures, 'ScatterPlot2D_LowFreq-REM_axes.png'))

    axes = [1, 2] # High-freq axis & REM axis
    points_prj = points_active[:, np.r_[axes]]
    mm = np.array([m[np.r_[axes]] for m in c_means[np.r_[0,1]]])
    cc = np.array([c[np.r_[axes]][:,np.r_[axes]] for c in c_covars[np.r_[0,1]]])
    fig = plot_scatter2D(points_prj, pred_active, mm , cc, colors, YLABEL, ZLABEL)
    fig.savefig(os.path.join(path2figures,'ScatterPlot2D_HighFreq-REM_axes.png'))

    axes = [0, 1] # Low-freq axis & High-freq axis
    points_prj = stage_coord[:, np.r_[axes]]
    colors =  [COLOR_WAKE, COLOR_REM, COLOR_NREM]
    mm_proj = np.array([m[np.r_[axes]] for m in c_means[np.r_[0,1,2]]])
    cc_proj = np.array([c[np.r_[axes]][:,np.r_[axes]] for c in c_covars[np.r_[0,1,2]]])
    fig = plot_scatter2D(points_prj, c_pred3, mm_proj , cc_proj, colors, XLABEL, YLABEL, diag_line=True)
    fig.savefig(os.path.join(path2figures,'ScatterPlot2D_LowFreq-HighFreq_axes_Wake_REM_NREM.png'))

    colors =  [COLOR_WAKE, COLOR_REM, COLOR_NREM]
    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH, SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=-135)

    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-150, 150)
    ax.set_xlabel(XLABEL, fontsize=10, rotation = 0)
    ax.set_ylabel(YLABEL, fontsize=10, rotation = 0)
    ax.set_zlabel(ZLABEL, fontsize=10, rotation = 0)
    ax.tick_params(axis='both', which='major', labelsize=8)


    for c in set(c_pred3):
        t_points = stage_coord_expacti[c_pred3==c]
        ax.scatter3D(t_points[:,0], t_points[:,1], t_points[:,2], s=0.005, color=colors[c])

        ax.scatter3D(t_points[:,0], t_points[:,1], min(ax.get_zlim()), s=0.001, color='grey')
        ax.scatter3D(t_points[:,0], max(ax.get_ylim()), t_points[:,2], s=0.001, color='grey')
        ax.scatter3D(max(ax.get_xlim()), t_points[:,1], t_points[:,2], s=0.001, color='grey')

    fig.savefig(os.path.join(path2figures,'ScatterPlot3D.png'))


def main(data_dir, result_dir, pickle_input_data):
    """ main """

    exp_info_df = read_exp_info(data_dir)
    (epoch_num, sample_freq, exp_label, rack_label, start_datetime, end_datetime) = interpret_exp_info(exp_info_df)
 
    n_fft = int(256 * sample_freq/100) # assures frequency bins compatibe among different sampleling frequencies
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129) # same frequency bins given by signal.welch()
    bidx_sleep_freq = (freq_bins<20) # 52 bins
    bidx_active_freq = (freq_bins>30) # 52 bins
    bidx_theta_freq = (freq_bins>=4) & (freq_bins<10) # 15 bins
    bidx_delta_freq = (freq_bins<4) # 11 bins
    bidx_muscle_freq = (freq_bins>30) # 52 bins

    mouse_info_df = read_mouse_info(data_dir)
    for i, r in mouse_info_df.iterrows():
        device_id = r[0]
        mouse_group = r[1]
        mouse_id = r[2]
        dob = r[3]
        note = r[4]

        print_log(f'#######################################')
        print_log(f'#### [{i+1}] of mouse.info / devide_id: {device_id}')
        print_log(f'Reading voltages')
        print_log(f'Epoch num:{epoch_num} recorded at sampling frequency {sample_freq}')
        (eeg_vm_org, emg_vm_org, not_yet_pickled) = read_voltage_matrices(
            data_dir, device_id, sample_freq, EPOCH_LEN_SEC, epoch_num, start_datetime)

        if (pickle_input_data and not_yet_pickled):
            # if the command line argument has the optinal flag for pickling, pickle the voltage matrices
            pickle_voltage_matrices(eeg_vm_org, emg_vm_org, data_dir, device_id)


        # recover nans in the data if possible
        nan_ratio_eeg = np.apply_along_axis(et.patch_nan, 1, eeg_vm_org)
        nan_ratio_emg = np.apply_along_axis(et.patch_nan, 1, emg_vm_org)


        # exclude unrecoverable epochs as unknown
        bidx_unknown = np.apply_along_axis(np.any, 1, np.isnan(
            eeg_vm_org)) | np.apply_along_axis(np.any, 1, np.isnan(emg_vm_org))
        eeg_vm = eeg_vm_org[~bidx_unknown,:]
        emg_vm = emg_vm_org[~bidx_unknown,:]

        # make data comparable among different mice. Not necessary for staging,
        # but convenient for subsequnet analyses.
        eeg_vm_norm = (eeg_vm - np.mean(eeg_vm))/np.std(eeg_vm)
        emg_vm_norm = (emg_vm - np.mean(emg_vm))/np.std(emg_vm)

        # power-spectrum normalization of EEG and EMG
        spec_norm_eeg = spectrum_normalize(eeg_vm_norm, n_fft, sample_freq)
        spec_norm_emg = spectrum_normalize(emg_vm_norm, n_fft, sample_freq)
        psd_norm_mat_eeg = spec_norm_eeg['psd']
        psd_norm_mat_emg = spec_norm_emg['psd']

        # remove extreme powers
        extrp_ratio_eeg = np.apply_along_axis(remove_extreme_power, 1, psd_norm_mat_eeg)
        extrp_ratio_emg = np.apply_along_axis(remove_extreme_power, 1, psd_norm_mat_emg)

        # save the PSD matrices and associated factors for subsequent analyses
        ## set bidx_unknown; other factors were set by spectrum_normalize()
        spec_norm_eeg['bidx_unknown'] = bidx_unknown 
        spec_norm_emg['bidx_unknown'] = bidx_unknown
        pickle_powerspec_matrices(spec_norm_eeg, spec_norm_emg, bidx_unknown, result_dir, device_id)

        # spread epochs on the 3D (Low freq. x High freq. x REM metric) space
        psd_mat = np.concatenate([
            psd_norm_mat_eeg.reshape(*psd_norm_mat_eeg.shape, 1),
            psd_norm_mat_emg.reshape(*psd_norm_mat_emg.shape, 1)
        ], axis=2)
        stage_coord = np.array([(
            np.sum(y[bidx_sleep_freq, 0]), 
            np.sum(y[bidx_active_freq,0]),
            np.sum(y[bidx_theta_freq,0])-np.sum(y[bidx_delta_freq, 0])-np.sum(y[bidx_muscle_freq, 1])
        ) for y in psd_mat])
        ndata = len(stage_coord)

        # cancel the weight bias of active/NREM clusters
        cwb = cancel_weight_bias(stage_coord[:,0:2])
        stage_coord[:, 0:2] = cwb['stage_coord_2D']

        # run the classification process
        try:
            pred_2D, pred_2D_proba, means_2D, covars_2D, stage_coord_nremflat, pred_3D, pred_3D_proba, means_3D, covars_3D = classification_process(
                stage_coord)
        except ValueError:
            print_log('Encountered an unhandlable analytical error during the staging. Check the ' 
                'date validity of the mouse.')
            
            continue

        # output staging result
        stage_proba = np.zeros(3*epoch_num).reshape(epoch_num, 3)
        proba_REM  = pred_3D_proba[:, 1]
        proba_WAKE = pred_3D_proba[:, 0]
        proba_NREM = pred_3D_proba[:, 2]
        stage_proba[~bidx_unknown, 0] = proba_REM
        stage_proba[~bidx_unknown, 1] = proba_WAKE
        stage_proba[~bidx_unknown, 2] = proba_NREM

        stage_call = np.repeat('Unknown', epoch_num)
        stage_call[~bidx_unknown] =  np.array([STAGE_LABELS[y] for y in pred_3D])

        print_log(f'[{i+1}]:Devide ID: {device_id} REM:{1440*np.sum(stage_call=="REM")/ndata:.2f} '\
                f'NREM:{1440*np.sum(stage_call=="NREM")/ndata:.2f} '\
                f'Wake:{1440*np.sum(stage_call=="Wake")/ndata:.2f}')

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
        stage_file_path = os.path.join(result_dir, f'{device_id}.faster2.stage.csv')

        with open(stage_file_path, 'w', encoding='UTF-8') as f:
            f.write(f'# Exp label: {exp_label} Recorded at {rack_label}\n')
            f.write(f'# Device ID: {device_id} Mouse group: {mouse_group} Mouse ID: {mouse_id} DOB: {dob}\n')
            f.write(f'# Start: {start_datetime} End: {end_datetime} Note: {note}\n')
            f.write(f'# Epoch num: {epoch_num}\n')
            f.write(f'# Sampling frequency: {sample_freq}\n')
            f.write(f'# Staged by {FASTER2_NAME}\n')
        with open(stage_file_path, 'a') as f:
            stage_table.to_csv(f, header=True, index=False, line_terminator='\n')

        path2figures = os.path.join(result_dir, 'figure', f'{device_id}')
        os.makedirs(path2figures, exist_ok=True)

        # draw the bias histogram
        plot_hist_on_separation_axis(path2figures, cwb['proj_data'], cwb['means'], cwb['covars'], cwb['weights'])
        
        # draw scatter plots
        draw_scatter_plots(path2figures,  stage_coord, pred_2D,
                           means_2D, covars_2D, stage_coord_nremflat, pred_3D, means_3D, covars_3D)

        # pickle cluster parameters
        pickle_cluster_params(means_2D, covars_2D, means_3D, covars_3D, result_dir, device_id)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, help="path to the directory of input data")
    parser.add_argument("-r", "--result_dir", required=True, help="path to the directory of staging result")
    parser.add_argument("-p", "--pickle_input_data", help="flag to pickle input data", action='store_true')

    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    pickle_input_data = args.pickle_input_data

    os.makedirs(result_dir, exist_ok=True)
    dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log = initialize_logger(os.path.join(result_dir, f'stage.{dt_str}.log'))
    main(args.data_dir, result_dir, args.pickle_input_data)
