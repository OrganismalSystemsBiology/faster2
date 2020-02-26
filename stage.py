# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import eeg_tools as et
import re
from datetime import datetime, timedelta
from scipy import signal
from hmmlearn import hmm
import matplotlib as mpl
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from scipy import linalg

EPOCH_LEN_SEC = 8
STAGE_LABELS = ['REM', 'Wake', 'NREM']
XLABEL = 'Total low-freq. log-powers'
YLABEL = 'Total high-freq. log-powers'
ZLABEL = 'REM metric'
SCATTER_PLOT_FIG_WIDTH = 6   # inch
SCATTER_PLOT_FIG_HEIGHT = 6  # inch
FIG_DPI = 100 # dot per inch
COLOR_WAKE = '#EF5E26'
COLOR_NREM = '#23B4EF'
COLOR_REM  = 'olivedrab'

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

    csv_df = pd.read_csv(filepath,
                         engine="python",
                         dtype={'Device label': str, 'Mouse Group': str,
                                'Mouse ID': str, 'DOB': str, 'Note': str},
                         names=["Device label", "Mouse Group",
                                "Mouse ID", "DOB", "Note"],
                         skiprows=1,
                         header=None)

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


def read_voltage_matrices(data_dir, device_id, epoch_num, sample_freq, epoch_len_sec):
    """ This function reads dsi.txt files of EEG and EMG data, then returns matrices
    in the shape of (epochs, signals).
    
    Args:
        data_dir (str): a path to the dirctory that contains dsi.txt/ directory.
        device_id (str): a transmitter ID (e.g., ID47476) or channel ID (e.g., 09). 
        epoch_num (int): the number of epochs to be read.
        sample_freq (int): sampling frequency 
        epoch_len_sec (int): the length of an epoch in seconds
    
    Returns:
        [(np.array(2), np.arrray(2))]: a pair of voltage 2D matrices in a tuple
    """
    dsi_reader_eeg = et.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'), 
                                       f'{device_id}', 
                                       'EEG', 
                                       sample_freq=sample_freq)
    dsi_reader_emg = et.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'), 
                                       f'{device_id}', 
                                       'EMG', 
                                       sample_freq=sample_freq)
    
    eeg_df = dsi_reader_eeg.read_epochs(1, epoch_num)
    emg_df = dsi_reader_emg.read_epochs(1, epoch_num)

    eeg_vm = eeg_df.value.values.reshape(-1, epoch_len_sec * sample_freq)
    emg_vm = emg_df.value.values.reshape(-1, epoch_len_sec * sample_freq)

    return (eeg_vm, emg_vm)


def interpret_datetimestr(datetime_str):
    """ Convert datetime string to a datatime object allowing some variant forms
    
    Args:
        datetime_str (string): a string representing datetime
    
    Raises:
        ValueError: raised when interpretation is failed
    """

    datestr_patterns = [r'^(\d{4})(\d{2})(\d{2})',
                        r'^(\d{4})/(\d{1,2})/(\d{1,2})',
                        r'^(\d{4})-(\d{1,2})-(\d{1,2})']

    timestr_patterns = [r'(\d{2})(\d{2})(\d{2})$',
                        r'(\d{1,2}):(\d{1,2}):(\d{1,2})$',
                        r'(\d{1,2})-(\d{1,2})-(\d{1,2})$']

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
    start_datetime_str = exp_info_df['Start datetime'].values[0]
    end_datetime_str = exp_info_df['End datetime'].values[0]
    start_datetime = interpret_datetimestr(start_datetime_str)
    end_datetime = interpret_datetimestr(end_datetime_str)

    epoch_num = int((end_datetime - start_datetime).total_seconds() / EPOCH_LEN_SEC)

    sample_freq = exp_info_df['Sampling freq'].values[0]

    return (epoch_num, sample_freq)


def psd(y, n_fft, sample_freq):
    return signal.welch(y, nfft=n_fft, fs=sample_freq)[1][0:129]


def plot_scatter2D(points_2D, classes, means, covariances, colors, xlabel, ylabel):
    fig = Figure(figsize=(SCATTER_PLOT_FIG_WIDTH, SCATTER_PLOT_FIG_HEIGHT), dpi=FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, colors)):
        w, v = linalg.eigh(covar)
        w = 4. * np.sqrt(w) # 95% confidence (2SD) area

        ax.scatter(points_2D[classes == i, 0], points_2D[classes == i, 1], .01, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(v[0, 1] / v[0, 0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, w[0], w[1], 180. + angle, facecolor='none', edgecolor=color)
        ax.add_patch(ell)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    return fig


def shrink_rem_cluster(means, covar):
    """ By definition, REM epochs should not be z<0. This function focuses on the 
    ellipsoid representing the 95% confidence area of REM cluster. If this function  
    finds any ellipsoid axis penetrating the xy plane (i.e. the end-point is 
    at z<0), it shrinks the length of the axis to make the end-point be at z = 0.
    """
    z_mean = means[2]

    W, V = np.linalg.eigh(covar) # W: eigen values, V:eigen vectors (unit length)
    
    if np.any(z_mean - np.abs(V[2,:])*2*np.sqrt(W) <0):
        idx_of_maxZ = np.argmax(np.abs(V[2,:]*np.sqrt(W))) # find the maxZ-axis: an axis with the maximum abs(Z)

        new_w = (z_mean/(2*np.abs(V[2, idx_of_maxZ])))**2 # 2SD (95% confidnece area) of the new maxZ-axis points at z=0

        delta_w = W[idx_of_maxZ] - new_w

        zc = np.abs(V[2,:] / V[2, idx_of_maxZ]) # z contributions of all axes relative to the maxZ-axis
        W_updated = W - delta_w * zc 

        covar_updated = V@np.diag(W_updated)@V.T
    else:
        # if no axis found under z=0, return zero-size array
        covar_updated = np.array([])

    return covar_updated
 

def main(data_dir, result_dir):
    """ main """

    exp_info_df = read_exp_info(data_dir)
    (epoch_num, sample_freq) = interpret_exp_info(exp_info_df)
 
    n_fft = int(256 * sample_freq/100) # assures compatibe frequency bins regardless different sampleling frequencies
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129) # same frequency bins given by signal.welch()
    bidx_sleep_freq = (freq_bins<20)
    bidx_active_freq = (freq_bins>30)
    bidx_theta_freq = (freq_bins>4) & (freq_bins<10)
    bidx_delta_freq = (freq_bins<4)
    bidx_muscle_freq = (freq_bins>30)

    mouse_info_df = read_mouse_info(data_dir)
    for i, r in mouse_info_df.iterrows():
        device_id = r[0]

        print(f'[{i}] Reading voltages of {device_id}')
        print(f'epoch num:{epoch_num} recorded at sample frequency {sample_freq}')
        (eeg_vm, emg_vm) = read_voltage_matrices(data_dir, device_id, epoch_num, sample_freq, EPOCH_LEN_SEC)

        # replace nans in the data if possible
        nan_ratio_eeg = np.apply_along_axis(et.patch_nan, 1, eeg_vm)
        nan_ratio_emg = np.apply_along_axis(et.patch_nan, 1, emg_vm)     

        # power-spectrum normalization of EEG
        psd_mat_eeg = np.apply_along_axis(lambda y: psd(y, n_fft, sample_freq), 1, eeg_vm)
        psd_mat_eeg = 10*np.log10(psd_mat_eeg) # decibel of power-spectrum-density (v**2/Hz)
        psd_mean_eeg  = np.apply_along_axis(np.nanmean, 0, psd_mat_eeg)
        psd_sd_eeg  = np.apply_along_axis(np.nanstd, 0, psd_mat_eeg)
        spec_norm_fac_eeg = 1/psd_sd_eeg
        psd_norm_mat_eeg = np.apply_along_axis(lambda y: spec_norm_fac_eeg*(y - psd_mean_eeg),
                                                   1,
                                                   psd_mat_eeg)

        # power-spectrum normalization of EMG
        psd_mat_emg = np.apply_along_axis(lambda y: psd(y, n_fft, sample_freq), 1, emg_vm)
        psd_mat_emg = 10*np.log10(psd_mat_emg) # decibel of power-spectrum-density (v**2/Hz)
        psd_mean_emg  = np.apply_along_axis(np.nanmean, 0, psd_mat_emg)
        psd_sd_emg  = np.apply_along_axis(np.nanstd, 0, psd_mat_emg)
        spec_norm_fac_emg = 1/psd_sd_emg
        psd_norm_mat_emg = np.apply_along_axis(lambda y: spec_norm_fac_emg*(y - psd_mean_emg),
                                                   1,
                                                   psd_mat_emg)

        # spread epochs on the 3D (active x sleep x REM) plane
        psd_mat = np.concatenate([
            psd_norm_mat_eeg.reshape(*psd_norm_mat_eeg.shape, 1),
            psd_norm_mat_emg.reshape(*psd_norm_mat_emg.shape, 1)
        ], axis=2)

        stage_coord = np.array([(
            np.sum(y[bidx_sleep_freq, 0]), 
            np.sum(y[bidx_active_freq,0]),
            np.sum(y[bidx_theta_freq,0])-np.sum(y[bidx_delta_freq, 0])-np.sum(y[bidx_muscle_freq, 1])
        ) for y in psd_mat])

        bidx_unknown = np.apply_along_axis(np.any, 1, np.isnan(stage_coord))
        stage_coord_valid = stage_coord[~bidx_unknown,:]
        ndata = len(stage_coord_valid)

        # classify active and sleep stages by Gaussian HMM on the 2D plane of (active x sleep)
        model = hmm.GaussianHMM(n_components=2, covariance_type='full', init_params='tc')
        model.startprob_ = np.array([0.5, 0.5])
        model.means_ = np.array([[-20, 20],[20, -20]])
        remodel = model.fit(stage_coord_valid[:, 0:2])
        print(remodel.means_)
        print(remodel.covars_)
        pred = remodel.predict(stage_coord_valid[:, 0:2])

        # leave only the active epochs expanded on z-axis (compress NREM epochs on xy-plane (z=0) )
        stage_coord_expacti = np.array([[y[0], y[1], y[2] if pred[i] == 0 else 0]
                                       for i, y in enumerate(stage_coord_valid)])

        # classify REM, Wake, and NREM (first classification)
        model = hmm.GaussianHMM(n_components=3, covariance_type='full', init_params='c', params='smtc')
        model.startprob_ = np.array([0.1, 0.45, 0.45])
        model.means_ = np.array([[-20, 10, 100], [-20, 20, -50], [20, -20, 0]])
        model.transmat_ = np.array([[8.76994067e-01, 6.53922215e-02, 5.76137113e-02],
                                    [7.45143157e-04, 9.68280746e-01, 3.09741107e-02],
                                    [1.00482802e-02, 2.49591897e-02, 9.64992530e-01]])

        remodel_active = model.fit(stage_coord_expacti)
        print(remodel_active.score(stage_coord_expacti))
        print(remodel_active.means_)
        print(remodel_active.covars_)

        pred3 = remodel_active.predict(stage_coord_expacti)
        print(f'REM:{1440*np.sum(pred3==0)/ndata} NREM:{1440*np.sum(pred3==2)/ndata} Wake:{1440*np.sum(pred3==1)/ndata}')

        covar_REM_updated = shrink_rem_cluster(remodel_active.means_[0], remodel_active.covars_[0])
        if covar_REM_updated.size > 0:
            # second classification with the updated covariance of the REM claster
            mm_old = np.copy(remodel_active.means_)
            cc_old = np.copy(remodel_active.covars_)
            cc_old[0] = covar_REM_updated

            model = hmm.GaussianHMM(n_components=3, covariance_type='full', init_params='', params='st')
            model.startprob_ = np.array([0.1, 0.45, 0.45])
            model.transmat_ = np.array([[8.76994067e-01, 6.53922215e-02, 5.76137113e-02],
                                        [7.45143157e-04, 9.68280746e-01, 3.09741107e-02],
                                        [1.00482802e-02, 2.49591897e-02, 9.64992530e-01]])
            model.means_ = mm_old
            model.covars_ = cc_old

            remodel_active = model.fit(stage_coord_expacti)
            print(remodel_active.score(stage_coord_expacti))
            print(remodel_active.means_)
            print(remodel_active.covars_)
        
            pred3 = remodel_active.predict(stage_coord_expacti)
            print(f'[UPDATED] REM:{1440*np.sum(pred3==0)/ndata} NREM:{1440*np.sum(pred3==2)/ndata} Wake:{1440*np.sum(pred3==1)/ndata}')
    
        # output staging result
        stage = np.repeat('Unknown', epoch_num)
        stage[~bidx_unknown] = np.array([STAGE_LABELS[p] for p in pred3])
        stage4csv = np.concatenate([np.repeat('#',7), stage])
        os.makedirs(result_dir, exist_ok=True)
        pd.DataFrame(stage4csv).to_csv(os.path.join(result_dir, f'{device_id}.auto.stage.csv'),
                                       header=False,
                                       index=False)

        # draw scatter plots
        path2figures = os.path.join(result_dir, 'figure', f'{device_id}')
        os.makedirs(path2figures, exist_ok=True)

        colors =  ['#EF5E26', '#23B4EF'] # Wake, NREM
        axes = [0, 1]
        points = stage_coord_valid[:, np.r_[axes]]
        fig = plot_scatter2D(points, pred, remodel.means_ , remodel.covars_, colors, XLABEL, YLABEL)
        fig.savefig(os.path.join(path2figures,'ScatterPlot2D_LowFreq-HighFreq_Axes.png'))

        points_active = stage_coord_expacti[((pred3==0) | (pred3==1)), :]
        pred_active = pred3[((pred3==0) | (pred3==1))]

        axes = [0, 2] # Low-freq axis & REM axis
        points_prj = points_active[:, np.r_[axes]]
        colors =  ['olivedrab', '#EF5E26', ] # REM, Wake
        mm = np.array([m[np.r_[axes]] for m in remodel_active.means_[np.r_[0,1]]])
        cc = np.array([c[np.r_[axes]][:,np.r_[axes]] for c in remodel_active.covars_[np.r_[0,1]]])
        fig = plot_scatter2D(points_prj, pred_active, mm , cc, colors, XLABEL, ZLABEL)
        fig.savefig(os.path.join(path2figures, 'ScatterPlot2D_LowFreq-REM_axes.png'))

        axes = [1, 2] # High-freq axis & REM axis
        points_prj = points_active[:, np.r_[axes]]
        mm = np.array([m[np.r_[axes]] for m in remodel_active.means_[np.r_[0,1]]])
        cc = np.array([c[np.r_[axes]][:,np.r_[axes]] for c in remodel_active.covars_[np.r_[0,1]]])
        fig = plot_scatter2D(points_prj, pred_active, mm , cc, colors, YLABEL, ZLABEL)
        fig.savefig(os.path.join(path2figures,'ScatterPlot2D_HighFreq-REM_axes.png'))

        colors =  ['olivedrab', '#EF5E26', '#23B4EF'] # REM, Wake, NREM
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


        for c in set(pred3):
            t_points = stage_coord_expacti[pred3==c]
            ax.scatter3D(t_points[:,0], t_points[:,1], t_points[:,2], s=0.005, color=colors[c])

            ax.scatter3D(t_points[:,0], t_points[:,1], min(ax.get_zlim()), s=0.001, color='grey')
            ax.scatter3D(t_points[:,0], max(ax.get_ylim()), t_points[:,2], s=0.001, color='grey')
            ax.scatter3D(max(ax.get_xlim()), t_points[:,1], t_points[:,2], s=0.001, color='grey')

        fig.savefig(os.path.join(path2figures,'ScatterPlot3D.png'))

    return 0

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--data_dir", help="path to the directory of input data")
   parser.add_argument("-r", "--result_dir", help="path to the directory of staging result")

   args = parser.parse_args()
   
   main(args.data_dir, args.result_dir)
