# -*- coding: utf-8 -*-
import matplotlib.backends.backend_pdf
import matplotlib._ttconv
from matplotlib.figure import Figure
import numpy as np
import stage
import pandas as pd
import faster2lib.eeg_tools as et

from datetime import timedelta
import os

class Timeseries_plot:
    def __init__(self, eeg_vm, emg_vm, stage_df, device_id, start_datetime, sample_freq):
        """This class handles drawing a set of timeseries plots of a mouse.
        
        Args:
            eeg_vm (np.array(2)): 2D array of EEG data
            emg_vm (np.array(2)): 2D array of EMG data
            stage_df (pd.DataFrame): a dataframe given by pd.read_csv(stage file)
            device_id (str): a string of device ID
            start_datetime (datetime): datetime of the first epoch
            sample_freq (int): sampling frequency

        Note: 
            # The length of a row is ~70 seconds
        """
        self.eeg_vm = eeg_vm
        self.emg_vm = emg_vm
        self.stage_df = stage_df
        self.device_id = device_id
        self.start_datetime = start_datetime
        self.sample_freq = sample_freq
        self.epoch_num = self.eeg_vm.shape[0]
        self.epoch_len_sec = int(self.eeg_vm.shape[1]/self.sample_freq)
        self.row_len_sec = row_len_sec(self.epoch_len_sec)
        self.page_num = int(np.ceil(self.epoch_num*self.epoch_len_sec/(self.row_len_sec*5))) 
        self.lines_eeg = []
        self.lines_emg = []
        self.lines_score = []
        self.texts_datetime = []
        self.texts_stage = []
        
        # initialize Figure
        self.fig=Figure(facecolor="w")
        self.axes = self._prepare_axes(self.fig)

        # set features for all axes
        for ax in self.axes.flatten():
            self._set_common_features(ax)

        # set features for the EEG/EMG axes
        for ax in self.axes[:, np.r_[0,1]].flatten():
            ax.set_ylim(-5, 5)
            ax.set_yticks([-3, 0, 3])

        for ax in self.axes[:, 0].flatten():
            ax.spines['top'].set_visible(True)

        # set features for the probability axes
        for ax in self.axes[:,2].flatten():
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0, 0.5, 1])

        self.fig.set_size_inches(20.64516129032258, 15.58687563423159) # 1600 x 1200 px
       
        # pad extra data to fill a page
        epoch_num_apage = int(5*self.row_len_sec/self.epoch_len_sec)
        if self.epoch_num % epoch_num_apage > 0:
            r = self.epoch_num % epoch_num_apage
            r_vm = np.zeros((r, self.epoch_len_sec * self.sample_freq))
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
        """ sets common features to axes.
        
        Args:
            ax (matplotlib.axes.Axes): axes given by add_subplot()
        """
        ax.set_xlim(0, self.row_len_sec)
        ax.grid(dashes=(2,2))
        ax.set_xticks(np.arange(0, self.row_len_sec, self.epoch_len_sec))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both', length=0, labelsize=0)


    def _prepare_axes(self, fig):
        """ prepares axes for plots
        
        Args:
            fig (matplotlib.figure.Figure): Figure object to contain axes
        
        Returns:
            [np.array(2)]: 2D (5x3) matrix of axes. 5 correspons to 5 rows in a page. 3 corresponds
            to EEG, EMG, and score plots, respectively. 
        """

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
        

    def _plot_a_row(self, row_index, x_eegemg, timestr, y_eeg, y_emg, p_rem, p_nrem, p_wake, epoch_nums, sleep_stages):
        """ draws a row of timeseries plots. 
        
        Args:
            row_index (int): index of the row to be drawn (from 0 to 4)
            x_eegemg (np.array(1)): x coordinates for EEG/EMG plots
            timestr (str): a string of datetime
            y_eeg (np.array(1)): y values of EEG plot
            y_emg (np.array(1)): y values of EMG plot
            p_rem (np.array(1)): epoch's probabilities of REM
            p_nrem (np.array(1)): epoch's probabilities of NREM
            p_wake (np.array(1)): epoch's probabilities of Wake
            epoch_nums (int): epoch's numbers of the row
            sleep_stages (np.array(1):str): a str array of sleep stages of the row
            epoch_len_sec: epoch length in seconds
        """
        axes = self.axes[row_index,:]
        epoch_num_arow = int(self.row_len_sec / self.epoch_len_sec)
        
        score_x = np.linspace(0, self.row_len_sec, 
                              50 * epoch_num_arow) # 50 points for an epoch.

        base_curve = np.ones(len(score_x))
        score_rem = base_curve*np.repeat(p_rem, 50) # 50 points for an epoch
        score_nrem = base_curve*np.repeat(p_nrem, 50)
        score_wake = base_curve*np.repeat(p_wake, 50)


        if len(self.lines_eeg)<=row_index:
            # initialize artists

            # plot graphs
            line_eeg, = axes[0].plot(x_eegemg, y_eeg, color='C0', linewidth=0.3)
            line_emg, = axes[1].plot(x_eegemg, y_emg, color='C3', linewidth=0.3)
            axes[2].plot(score_x, score_rem, color=stage.COLOR_REM, linewidth=1)
            axes[2].plot(score_x, score_nrem, color=stage.COLOR_NREM, linewidth=1)
            axes[2].plot(score_x, score_wake, color=stage.COLOR_WAKE, linewidth=1)
            line_scores = axes[2].get_lines()

            # plot texts
            txt_datetime = axes[0].text(0, 4.8, timestr, fontname='Arial', fontsize=11, ha='left', va='top')
            txt_stages = []
            font_size = 9 if self.epoch_len_sec < 8 else 11
            for i,s in enumerate(sleep_stages):
                txt_stage = axes[0].text(i*self.epoch_len_sec, 3.2, f'{epoch_nums[i]}: {s}', fontname='Arial', fontsize=font_size, ha='left', va='top')
                txt_stages.append(txt_stage)

            # store artists for reuse
            self.lines_eeg.append(line_eeg)
            self.lines_emg.append(line_emg)
            self.lines_score.append(line_scores)
            self.texts_datetime.append(txt_datetime)
            self.texts_stage.append(txt_stages)
        else:
            # update artists
            self.lines_eeg[row_index].set_ydata(y_eeg)
            self.lines_emg[row_index].set_ydata(y_emg)
            self.lines_score[row_index][0].set_ydata(score_rem)
            self.lines_score[row_index][1].set_ydata(score_nrem)
            self.lines_score[row_index][2].set_ydata(score_wake)
            self.texts_datetime[row_index].set_text(timestr)
            for i, s in enumerate(sleep_stages):
                self.texts_stage[row_index][i].set_text(f'{epoch_nums[i]}: {s}')
                
        return


    def plot_timeseries_a_page(self, page):
        """ draws a page of timeseries plots and save a file. A page contains 5 rows.
        A row contains ~70 seconds of data (72 seconds: 9 epochs when epoch_len_sec=8).
       
        Args:
            page (int): A page number to be drawn
        """
        epp = int(5 * self.row_len_sec/self.epoch_len_sec) # epochs per page
        y_eeg = self.eeg_vm[(page-1)*epp:(page)*epp,:].flatten()
        y_emg = self.emg_vm[(page-1)*epp:(page)*epp,:].flatten()
        p_rem  = self.stage_df.iloc[:,1].values[(page-1)*epp:(page)*epp]
        p_nrem = self.stage_df.iloc[:,2].values[(page-1)*epp:(page)*epp]
        p_wake = self.stage_df.iloc[:,3].values[(page-1)*epp:(page)*epp]
        stages = self.stage_df.iloc[:,0].values[(page-1)*epp:(page)*epp]
        epoch_nums = range((page-1)*epp+1, (page)*epp+1)
        timestamps = [self.start_datetime + timedelta(seconds=(page-1)*5*self.row_len_sec + i*self.row_len_sec) for i in range(5)]
        
        x = np.linspace(0, self.row_len_sec, self.row_len_sec*self.sample_freq)
        epr = int(self.row_len_sec/self.epoch_len_sec) # epochs per row
        for i in range(5):
            ts = timestamps[i].strftime("%Y/%m/%d %H:%M:%S")
            y_ee = y_eeg[i*self.row_len_sec*self.sample_freq:((i+1)*self.row_len_sec)*self.sample_freq] 
            y_em = y_emg[i*self.row_len_sec*self.sample_freq:((i+1)*self.row_len_sec)*self.sample_freq]
            p_r = p_rem[i*epr:(i+1)*epr]
            p_n = p_nrem[i*epr:(i+1)*epr]
            p_w = p_wake[i*epr:(i+1)*epr]
            en = epoch_nums[i*epr:(i+1)*epr]
            ss = stages[i*epr:(i+1)*epr]

            self._plot_a_row(i, x, ts, y_ee, y_em, p_r, p_n, p_w, en, ss)
        
        self.fig.canvas.draw()
        filename = f'{self.device_id}.{epoch_nums[0]:06}.jpg'

        self.fig.savefig(filename, pad_inches=0, bbox_inches='tight', dpi=100, pil_kwargs={'quality':85, 'optimize':True})


def plot_timeseries_a_mouse(voltage_data_dir, stage_dir, result_dir, device_id, sample_freq, epoch_num, epoch_len_sec, start_datetime):
    """ wraps the process to draw a set of plots of a mouse over epochs. 
    This function is convenient to run a drawing process in multiprocessing.
    
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
    try:
        stage_df = pd.read_csv(stage_filepath, skiprows=7,
                            header=None, engine='python')
    except FileNotFoundError:
        print(f'Plotting without stage info. FileNotFound: {stage_filepath}.')
        # serve a dummy stage information
        stage_df = pd.DataFrame({'Stage': np.repeat('', epoch_num),
                                 'REM probability': np.zeros(epoch_num),
                                 'NREM probability': np.zeros(epoch_num),
                                 'Wake probability': np.zeros(epoch_num),
                                 'NaN ratio EEG-TS': np.zeros(epoch_num),
                                 'NaN ratio EMG-TS': np.zeros(epoch_num),
                                 'Outlier ratio EEG-TS': np.zeros(epoch_num),
                                 'Outlier ratio EMG-TS': np.zeros(epoch_num)})

    if len(stage_df) != epoch_num:
        raise(ValueError('Stage length is not consistent with the epoch number.'))

    (eeg_vm_org, emg_vm_org, _) = et.read_voltage_matrices(voltage_data_dir, device_id, sample_freq, epoch_len_sec, epoch_num, start_datetime)
    eeg_vm_norm = stage.voltage_normalize(eeg_vm_org)
    emg_vm_norm = stage.voltage_normalize(emg_vm_org)

    tp = Timeseries_plot(eeg_vm_norm, emg_vm_norm, stage_df,
                         device_id, start_datetime, sample_freq)


    plot_dir = os.path.join(result_dir, 'figure', 'voltage', f'{device_id}_tmp')
    os.makedirs(plot_dir, exist_ok=True)
    os.chdir(plot_dir)
    print(f'Drawing plots in: {plot_dir}')
    for i in range(1, tp.page_num + 1):
        tp.plot_timeseries_a_page(i)
    
    # move to the upper level (to release the lock of the directory)
    os.chdir('..')


def row_len_sec(epoch_len_sec):
    """The formula for the row length in seconds
    The row length is a multiple of the epoch length and around 70 seconds

    Args:
        epoch_len_sec (int): The 

    Returns:
        row_len_sec: The row length in seconds
    """
    return int(epoch_len_sec * round(70/epoch_len_sec))
    
