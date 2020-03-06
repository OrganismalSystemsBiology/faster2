# -*- coding: utf-8 -*-
import matplotlib
from matplotlib.figure import Figure
import numpy as np
import stage
import pandas as pd

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class Timeseries_plot:
    def __init__(self, eeg_vm, emg_vm, stage_df, device_id, start_datetime, sample_freq):
        self.eeg_vm = eeg_vm
        self.emg_vm = emg_vm
        self.stage_df = stage_df
        self.device_id = device_id
        self.start_datetime = start_datetime
        self.sample_freq = sample_freq
        self.epoch_num = self.eeg_vm.shape[0]
        self.page_num = int(np.ceil(self.epoch_num/45)+1)

        if self.epoch_num % 45 > 0:
            r = self.epoch_num % 45
            r_vm = np.zeros((r, 8*self.sample_freq))
            self.eeg_vm = np.vstack([eeg_vm, r_vm])
            self.emg_vm = np.vstack([emg_vm, r_vm])

            stage_table = pd.DataFrame({0: np.repeat('empty', r),
                                        1: np.zeros(r),
                                        2: np.zeros(r),
                                        3: np.zeros(r),
                                        4: np.zeros(r),
                                        5: np.zeros(r),
                                        6: np.zeros(r),
                                        7: np.zeros(r)})
            
            self.stage_df = pd.concat([stage_df, stage_table], ignore_index=True)


    def _set_common_features(self, ax):
        ax.set_xlim(0, 72)
        ax.grid(dashes=(2,2))
        ax.set_xticks(np.arange(0, 72, 8))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both', length=0, labelsize=0)


    def _prepare_axes(self, fig):
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        gs = fig.add_gridspec(nrows = 7*5, ncols=1)
        ax1_1 = fig.add_subplot(gs[0:3, :], xmargin=0, ymargin=0)
        ax1_2 = fig.add_subplot(gs[3:6, :], xmargin=0, ymargin=0)
        ax1_3 = fig.add_subplot(gs[6:7, :], xmargin=0, ymargin=0)

        ax2_1 = fig.add_subplot(gs[7:10, :], xmargin=0, ymargin=0)
        ax2_2 = fig.add_subplot(gs[10:13, :], xmargin=0, ymargin=0)
        ax2_3 = fig.add_subplot(gs[13:14, :], xmargin=0, ymargin=0)

        ax3_1 = fig.add_subplot(gs[14:17, :], xmargin=0, ymargin=0)
        ax3_2 = fig.add_subplot(gs[17:20, :], xmargin=0, ymargin=0)
        ax3_3 = fig.add_subplot(gs[20:21, :], xmargin=0, ymargin=0)

        ax4_1 = fig.add_subplot(gs[21:24, :], xmargin=0, ymargin=0)
        ax4_2 = fig.add_subplot(gs[24:27, :], xmargin=0, ymargin=0)
        ax4_3 = fig.add_subplot(gs[27:28, :], xmargin=0, ymargin=0)

        ax5_1 = fig.add_subplot(gs[28:31, :], xmargin=0, ymargin=0)
        ax5_2 = fig.add_subplot(gs[31:34, :], xmargin=0, ymargin=0)
        ax5_3 = fig.add_subplot(gs[34:35, :], xmargin=0, ymargin=0)

        axes = np.array([[ax1_1, ax1_2, ax1_3],
                [ax2_1, ax2_2, ax2_3],
                [ax3_1, ax3_2, ax3_3],
                [ax4_1, ax4_2, ax4_3],
                [ax5_1, ax5_2, ax5_3]])
        
        return axes


    def _plot_a_row(self, axes_for_a_row, x_eegemg, timestr, y_eeg, y_emg, p_rem, p_nrem, p_wake, epoch_nums, sleep_stages):
        
        axes = axes_for_a_row
        
        score_x = np.linspace(0, 72, 450)
        base_curve = base_curve = np.ones(len(score_x))
        score_rem = base_curve*np.repeat(p_rem, 50)
        score_nrem = base_curve*np.repeat(p_nrem, 50)
        score_wake = base_curve*np.repeat(p_wake, 50)
        
        # plot
        axes[0].plot(x_eegemg, y_eeg, color='C0', linewidth=0.3)
        axes[1].plot(x_eegemg, y_emg, color='C3', linewidth=0.3)
        axes[2].plot(score_x, score_rem, color=stage.COLOR_REM, linewidth=1)
        axes[2].plot(score_x, score_nrem, color=stage.COLOR_NREM, linewidth=1)
        axes[2].plot(score_x, score_wake, color=stage.COLOR_WAKE, linewidth=1)

        # plot stage
        axes[0].text(0, 4.8, timestr, fontname='Arial', fontsize=11, ha='left', va='top')
        for i,s in enumerate(sleep_stages):
            axes[0].text(i*8, 3.2, f'{epoch_nums[i]}: {s}', fontname='Arial', fontsize=11, ha='left', va='top')


    def plot_timeseries_a_page(self, page):
        y_eeg = self.eeg_vm[(page-1)*45:(page)*45,:].flatten()
        y_emg = self.emg_vm[(page-1)*45:(page)*45,:].flatten()
        p_rem  = self.stage_df.iloc[:,1].values[(page-1)*45:(page)*45]
        p_nrem = self.stage_df.iloc[:,2].values[(page-1)*45:(page)*45]
        p_wake = self.stage_df.iloc[:,3].values[(page-1)*45:(page)*45]
        stages = self.stage_df.iloc[:,0].values[(page-1)*45:(page)*45]
        epoch_nums = range((page-1)*45+1, (page)*45+1)
        timestamps = [self.start_datetime + timedelta(seconds=(page-1)*45*8 + i*9*8) for i in range(5)]
        
        fig=Figure(facecolor="w")
        axes = self._prepare_axes(fig)

        # set features for all axes
        for ax in axes.flatten():
            self._set_common_features(ax)

        # set features for the EEG/EMG axes
        for ax in axes[:, np.r_[0,1]].flatten():
            ax.set_ylim(-5, 5),
            ax.set_yticks([-3, 0, 3])

        for ax in axes[:, 0].flatten():
            ax.spines['top'].set_visible(True)

        # set features for the probability axes
        for ax in axes[:,2].flatten():
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0,0.5,1])
        
        x = np.linspace(0, 72, 72*self.sample_freq)
        
        for i in range(5):
            ts = timestamps[i].strftime("%Y/%m/%d %H:%M:%S")
            y_ee = y_eeg[i*72*self.sample_freq:((i+1)*72)*self.sample_freq] 
            y_em = y_emg[i*72*self.sample_freq:((i+1)*72)*self.sample_freq]
            p_r = p_rem[i*9:(i+1)*9]
            p_n = p_nrem[i*9:(i+1)*9]
            p_w = p_wake[i*9:(i+1)*9]
            en = epoch_nums[i*9:(i+1)*9]
            ss = stages[i*9:(i+1)*9]

            self._plot_a_row(axes[i,:], x, ts, y_ee, y_em, p_r, p_n, p_w, en, ss)

        fig.set_size_inches(20.64516129032258, 15.58687563423159) # 1600 x 1200 px
        filename = f'{self.device_id}.{epoch_nums[0]:06}.jpg'

        fig.savefig(filename, pad_inches=0, bbox_inches='tight', dpi=100, quality=85, optimize=True)
        print(filename)
 

def plot_timeseries_a_mouse(voltage_data_dir, stage_dir, result_dir, device_id, sample_freq, epoch_num, start_datetime):
    """ wraps the process to draw a set of plots of a mouse over epochs. 
    
    Args:
        voltage_data_dir (str): full path to the directory of voltage data
        stage_dir (str): full path to the directory of stage data
        result_dir (str): full path to the directory of resulting plots
        device_id (str): device id to be plotted
        sample_freq (str): sampling frequency
        epoch_num (int): number of epochs to be plotted
        start_datetime (datetime): datetime of the first epoch 
    """
    
    stage_filepath = os.path.join(stage_dir, f'{device_id}.faster2.stage.csv')
    stage_df = pd.read_csv(stage_filepath, skiprows=7,
                           header=None, engine='python')

    (eeg_vm_org, emg_vm_org, _) = stage.read_voltage_matrices(voltage_data_dir, device_id, epoch_num, sample_freq, stage.EPOCH_LEN_SEC)
    eeg_vm_norm = (eeg_vm_org - np.nanmean(eeg_vm_org))/np.nanstd(eeg_vm_org)
    emg_vm_norm = (emg_vm_org - np.nanmean(emg_vm_org))/np.nanstd(emg_vm_org)

    tp = Timeseries_plot(eeg_vm_norm, emg_vm_norm, stage_df,
                         device_id, start_datetime, sample_freq)


    plot_dir = os.path.join(result_dir, 'figure', 'voltage', device_id)
    os.makedirs(plot_dir, exist_ok=True)
    os.chdir(plot_dir)
    for i in range(1, tp.page_num):
        tp.plot_timeseries_a_page(i)
