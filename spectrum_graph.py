# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
from scipy import linalg

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import stage


def log_psd_inv(y, normalizing_fac, normalizing_mean):
    """ inverses the spectrum normalization to get the conventional PSD
    """
    return 10**((y / normalizing_fac + normalizing_mean) / 10)


class SpectrumAnalysisPlots:
    def __init__(self, spec_norm_eeg, spec_norm_emg, cluster_params, device_id, sample_freq):
        """This class handles drawing a set of epoch's powerspectrum of a mouse.

        Args:
            spec_norm_eeg (np.array(epoch_num, freq_bins)): a dict saved by 
                stage.pickle_powerspec_matrices()
            spec_norm_emg (np.array(epoch_num, freq_bins)): a dict saved by 
                stage.pickle_powerspec_matrices()
            cluster_params: a dict of ['2stage-means', '2stage-covars', '3stage-means', 
                '3stage-covars']
            device_id (str): a string of device ID
            sample_freq (int): sampling frequency
        """
        self.eeg_norm_mat = spec_norm_eeg['psd']
        nf = spec_norm_eeg['norm_fac']
        nm = spec_norm_eeg['mean']
        self.eeg_conv_mat = np.vectorize(log_psd_inv)(self.eeg_norm_mat, nf, nm)
        self.eeg_conv_mean = np.apply_along_axis(np.mean, 0, self.eeg_conv_mat)
        self.eeg_conv_std = np.apply_along_axis(np.std, 0, self.eeg_conv_mat)
        self.eeg_norm_mean = np.apply_along_axis(np.mean, 0, self.eeg_norm_mat)
        self.eeg_norm_std = np.apply_along_axis(np.std, 0, self.eeg_norm_mat)

        self.emg_norm_mat = spec_norm_emg['psd']
        nf = spec_norm_emg['norm_fac']
        nm = spec_norm_emg['mean']
        self.emg_conv_mat = np.vectorize(log_psd_inv)(self.emg_norm_mat, nf, nm)
        self.emg_conv_mean = np.apply_along_axis(np.mean, 0, self.emg_conv_mat)
        self.emg_conv_std = np.apply_along_axis(np.std, 0, self.emg_conv_mat)
        self.emg_norm_mean = np.apply_along_axis(np.mean, 0, self.emg_norm_mat)
        self.emg_norm_std = np.apply_along_axis(np.std, 0, self.emg_norm_mat)

        self.device_id = device_id
        self.sample_freq = sample_freq
        self.epoch_num = self.eeg_norm_mat.shape[0]

        self.line_eeg_spec_conv = None
        self.line_emg_spec_conv = None
        self.line_eeg_spec_norm = None
        self.line_emg_spec_norm = None
        self.point_HLplane = None
        self.point_RLplane = None
        self.line_delta = None
        self.line_theta = None
        self.line_tdr = None

        # assures frequency bins compatibe among different sampleling frequencies
        n_fft = int(256 * sample_freq/100)
        # same frequency bins given by signal.welch()
        freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)
        bidx_sleep_freq = (freq_bins < 20)  # 52 bins
        bidx_active_freq = (freq_bins > 30)  # 52 bins
        bidx_theta_freq = (freq_bins >= 4) & (freq_bins < 10)  # 15 bins
        bidx_delta_freq = (freq_bins < 4)  # 11 bins
        bidx_muscle_freq = (freq_bins > 30)  # 52 bins

        # normalized power
        self.npower_sleep = np.apply_along_axis(
            np.sum, 1, self.eeg_norm_mat[:, bidx_sleep_freq])
        self.npower_active = np.apply_along_axis(
            np.sum, 1, self.eeg_norm_mat[:, bidx_active_freq])
        self.npower_muscle = np.apply_along_axis(
            np.sum, 1, self.emg_norm_mat[:, bidx_muscle_freq])
        self.npower_theta = np.apply_along_axis(
            np.sum, 1, self.eeg_norm_mat[:, bidx_theta_freq])
        self.npower_delta = np.apply_along_axis(
            np.sum, 1, self.eeg_norm_mat[:, bidx_delta_freq])
        # conventional power
        self.cpower_theta = np.apply_along_axis(
            np.sum, 1, self.eeg_conv_mat[:, bidx_theta_freq])
        self.cpower_delta = np.apply_along_axis(
            np.sum, 1, self.eeg_conv_mat[:, bidx_delta_freq])

        # initialize Figure
        self.fig = Figure(figsize=(12, 4), dpi=stage.FIG_DPI, facecolor="w")
        self.axes = self._prepare_axes(self.fig)

        # set features of ax: specturms
        self.axes[0].set_ylim(0, 0.4)
        self.axes[0].set_xticks([0, 5, 10, 20, 30, 40, 50])
        self.axes[2].set_ylim(-4, 4)
        self.axes[2].set_xticks([0, 5, 10, 20, 30, 40, 50])
        self.axes[1].set_ylim(0, 0.15)
        self.axes[1].set_xticks([0, 5, 10, 20, 30, 40, 50])
        self.axes[3].set_ylim(-4, 4)
        self.axes[3].set_xticks([0, 5, 10, 20, 30, 40, 50])

        # set feature of ax: clustermap low-high plane
        self.axes[4].set_xlim(-150, 150)
        self.axes[4].set_ylim(-150, 150)
        self.axes[4].set_yticklabels([])
        self.axes[4].set_xticklabels([])

        # set feature of ax: clustermap low-REM plane
        self.axes[5].set_xlim(-150, 150)
        self.axes[5].set_ylim(-150, 150)
        self.axes[5].set_yticklabels([])
        self.axes[5].set_xticklabels([])

        # set feature of ax: power timeseries
        self.axes[6].set_ylim(0, 4)

        # initialize plots
        (self.line_eeg_spec_conv, self.line_emg_spec_conv, self.line_eeg_spec_norm, self.line_emg_spec_norm) = self._initialize_spectrumplots(freq_bins, self.eeg_conv_mean, self.eeg_conv_std, self.emg_conv_mean, self.emg_conv_std,
                                                                                                                                              self.eeg_norm_mean, self.eeg_norm_std, self.emg_norm_mean, self.emg_norm_std,
                                                                                                                                              self.axes[0], self.axes[1], self.axes[2], self.axes[3])
        (self.point_HLplane, self.point_LRplane) = self._initialize_clustermaps(
            cluster_params, self.axes[4], self.axes[5])
        (self.line_delta, self.line_theta,
         self.line_tdr) = self._initialize_power_timeseries(self.axes[6])

    def _initialize_spectrumplots(self, freq_bins, eeg_conv_mean, eeg_conv_std, emg_conv_mean, emg_conv_std,
                                  eeg_norm_mean, eeg_norm_std, emg_norm_mean, emg_norm_std,
                                  ax1, ax2, ax3, ax4):
        # set reference values: EEG_spectrum_conv
        ax1.plot(freq_bins, self.eeg_conv_mean, color='black')
        ax1.fill_between(freq_bins, self.eeg_conv_mean - self.eeg_conv_std,
                         self.eeg_conv_mean + self.eeg_conv_std, color='black', alpha=0.3)

        # set reference values: EMG_spectrum_conv
        ax2.plot(freq_bins, self.emg_conv_mean, color='black')
        ax2.fill_between(freq_bins, self.emg_conv_mean - self.emg_conv_std,
                         self.emg_conv_mean + self.emg_conv_std, color='black', alpha=0.3)

        # set reference values: EEG_spectrum_norm
        ax3.plot(freq_bins, self.eeg_norm_mean, color='black')
        ax3.fill_between(freq_bins, self.eeg_norm_mean - self.eeg_norm_std,
                         self.eeg_norm_mean + self.eeg_norm_std, color='black', alpha=0.3)

        # set reference values: EMG_spectrum_norm
        ax4.plot(freq_bins, self.emg_norm_mean, color='black')
        ax4.fill_between(freq_bins, self.emg_norm_mean - self.emg_norm_std,
                         self.emg_norm_mean + self.emg_norm_std, color='black', alpha=0.3)

        line_eeg_spec_conv, = ax1.plot(freq_bins, np.zeros(len(freq_bins)))
        line_emg_spec_conv, = ax2.plot(freq_bins, np.zeros(len(freq_bins)))
        line_eeg_spec_norm, = ax3.plot(freq_bins, np.zeros(len(freq_bins)))
        line_emg_spec_norm, = ax4.plot(freq_bins, np.zeros(len(freq_bins)))

        return (line_eeg_spec_conv, line_emg_spec_conv, line_eeg_spec_norm, line_emg_spec_norm)

    def _initialize_clustermaps(self, cluster_params, ax1, ax2):
        # set reference ellipsoids in clustermap: High-low freq plane
        means = cluster_params['2stage-means']
        covars = cluster_params['2stage-covars']
        colors = [stage.COLOR_WAKE, stage.COLOR_NREM]
        labels = ['Active', 'NREM']
        for (mean, covar, color, label) in zip(means, covars, colors, labels):
            w, v = linalg.eigh(covar)
            w = 4. * np.sqrt(w)  # 95% confidence (2SD) area

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(v[0, 1] / v[0, 0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, w[0], w[1], 180. + angle,
                                      facecolor='none', edgecolor=color, label=label)
            ax1.add_patch(ell)
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,
                   fontsize=8)

        # set reference ellipsoids in clustermap: Low-REM plane
        c_means = cluster_params['3stage-means']
        c_covars = cluster_params['3stage-covars']
        axes = [0, 2]  # Low-freq axis & REM axis
        colors = [stage.COLOR_WAKE, stage.COLOR_REM]
        means = np.array([m[np.r_[axes]] for m in c_means[np.r_[0, 1]]])
        covars = np.array([c[np.r_[axes]][:, np.r_[axes]]
                           for c in c_covars[np.r_[0, 1]]])
        colors = [stage.COLOR_REM, stage.COLOR_WAKE]
        labels = ['REM', 'WAKE']
        for (mean, covar, color, label) in zip(means, covars, colors, labels):
            w, v = linalg.eigh(covar)
            w = 4. * np.sqrt(w)  # 95% confidence (2SD) area

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(v[0, 1] / v[0, 0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, w[0], w[1], 180. + angle,
                                      facecolor='none', edgecolor=color, label=label)
            ax2.add_patch(ell)
        ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,
                   fontsize=8)

        point_HLplane, = ax1.plot(0, 0, 'ro', color='C3')
        point_LRplane, = ax2.plot(0, 0, 'ro', color='C3')

        return (point_HLplane, point_LRplane)

    def _initialize_power_timeseries(self, ax):
        #set initial values in
        x_pos = np.arange(-45, 45)
        x_labels = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
        ax.set_xticks(x_pos, minor=True)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)
        # delta
        line_delta, = ax.plot(x_pos, np.zeros(
            90), color='C0', alpha=0.6, label=r'$\delta$')
        # theta
        line_theta, = ax.plot(x_pos, np.zeros(
            90), color='C1', alpha=0.6, label=r'$\theta$')
        # ratio of theta/delta
        line_tdr, = ax.plot(x_pos, np.zeros(
            90), color=stage.COLOR_REM, linewidth=2, label=r'$\delta/\theta$')
        ax.axvline(x=0)
        ax.set_ylabel('power')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=8)

        return(line_delta, line_theta, line_tdr)

    def _prepare_axes(self, fig):
        """ prepares axes for plots

        Args:
            fig (matplotlib.figure.Figure): Figure object to contain axes

        Returns:
            [np.array(7)]: 1D 7 matrices of axes. 7 correspons to 1. eeg_spec_conv,
            2. emg_spec_conv, 3. eeg_spec_norm, 4. emg_spec_norm, 5. high-low freq 
            clustermap, 6. low-REM clustermap, and 7. power timeseries
        """

        fig.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=0.2, hspace=1.6)
        gs = fig.add_gridspec(nrows=8, ncols=4)
        ax1 = fig.add_subplot(gs[0:3, 0], xmargin=0, ymargin=0)
        ax2 = fig.add_subplot(gs[3:6, 0], xmargin=0, ymargin=0)
        ax3 = fig.add_subplot(gs[0:3, 1], xmargin=0, ymargin=0)
        ax4 = fig.add_subplot(gs[3:6, 1], xmargin=0, ymargin=0)
        ax5 = fig.add_subplot(gs[0:6, 2], xmargin=0, ymargin=0)
        ax6 = fig.add_subplot(gs[0:6, 3], xmargin=0, ymargin=0)
        ax7 = fig.add_subplot(gs[6:8, :], xmargin=0, ymargin=0)

        return([ax1, ax2, ax3, ax4, ax5, ax6, ax7])

    def plot_specs_an_epoch(self, epoch_idx):
        e = epoch_idx - 1
        self.line_eeg_spec_conv.set_ydata(self.eeg_conv_mat[e, :])
        self.line_emg_spec_conv.set_ydata(self.emg_conv_mat[e, :])
        self.line_eeg_spec_norm.set_ydata(self.eeg_norm_mat[e, :])
        self.line_emg_spec_norm.set_ydata(self.emg_norm_mat[e, :])

        self.point_HLplane.set_xdata(self.npower_sleep[e])
        self.point_HLplane.set_ydata(self.npower_active[e])
        self.point_LRplane.set_xdata(self.npower_sleep[e])
        self.point_LRplane.set_ydata(
            self.npower_theta[e] - self.npower_delta[e] - self.npower_muscle[e])

        e_range = slice(max(e - 45, 0), min(e + 45, self.epoch_num))
        pad_l = max(0, 45 - e)
        pad_r = max(0, e + 45 - self.epoch_num)
        d = self.cpower_delta[e_range]
        t = self.cpower_theta[e_range]
        self.line_delta.set_ydata(np.pad(d, [pad_l, pad_r]))
        self.line_theta.set_ydata(np.pad(t, [pad_l, pad_r]))
        self.line_tdr.set_ydata(np.pad(t/d, [pad_l, pad_r]))

        self.fig.canvas.draw()
        filename = f'{self.device_id}.{epoch_idx:06}.jpg'
        self.fig.savefig(filename, dpi=100, quality=85, optimize=True)


def plot_specs_a_mouse(psd_data_dir, cluster_param_dir, result_dir, device_label, sample_freq):
    """ wraps the process to draw a set of spectrums of a mouse over epochs. 
    This function is convenient to run a drawing process in multiprocessing.
    
    Args:
        psd_data_dir (str): full path to the directory of the pickled PSD data
        cluster_param_dir (str): full path to the directory of the pickled cluster parametes
        result_dir (str): full path to the directory of resulting plots
        device_id (str): device id to be plotted
        sample_freq (str): sampling frequency
    """
    # read the normalized EEG PSDs and the associated normalization factors and means
    pkl_path_eeg = os.path.join(psd_data_dir, f'{device_label}_EEG_PSD.pkl')
    with open(pkl_path_eeg, 'rb') as pkl:
        spec_norm_eeg = pickle.load(pkl)
    
    # read the normalized EMG PSDs and the associated normalization factors and means
    pkl_path_emg = os.path.join(psd_data_dir, f'{device_label}_EMG_PSD.pkl')
    with open(pkl_path_emg, 'rb') as pkl:
        spec_norm_emg = pickle.load(pkl)

    pkl_path_clust = os.path.join(cluster_param_dir, f'{device_label}_cluster_params.pkl')
    with open(pkl_path_clust, 'rb') as pkl:
        cluster_params = pickle.load(pkl)

    sap = SpectrumAnalysisPlots(spec_norm_eeg, spec_norm_emg, cluster_params, device_label, sample_freq)

    root_plot_dir = os.path.join(result_dir, 'figure', 'spectrum', device_label)
    os.makedirs(root_plot_dir, exist_ok=True)
    for i in range(1, sap.epoch_num):
        # make subdirs for every 1000 epochs
        if (i-1) % 1000 == 0:
            plot_dir = os.path.join(root_plot_dir, f'{int(i/1000):06}')
            os.makedirs(plot_dir, exist_ok=True)
            os.chdir(plot_dir)
        
        sap.plot_specs_an_epoch(i)