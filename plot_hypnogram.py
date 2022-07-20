# -*- coding: utf-8 -*-
import os
import argparse
import pickle
from datetime import datetime
import numpy as np
from matplotlib.figure import Figure
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

import stage
import faster2lib.eeg_tools as et

# prefixed parameters
FIG_DPI = 300
COLOR_LIST = np.array([stage.COLOR_WAKE, stage.COLOR_NREM, stage.COLOR_REM])
LINE_W = 72/FIG_DPI
AXIS_W = 2*72/FIG_DPI


def remove_unknown_stage(stages):
    """ Removes UNKNOWN stages by extrapolating
    the previous stage.

    Arguments:
        stages: a numpy array of regularized stages.
        Note this function works in the destructive way.

    Returns:
        A numpy array of stages without UNKNOWN.

    """

    idx_unknown = np.where(stages == 'UNKNOWN')[0]

    # The first entry cannot be UNKNOWN
    if stages[0] == 'UNKNOWN':
        not_unknown = [stg for stg in stages if stg != 'UNKNOWN']
        if len(not_unknown) == 0:
            raise ValueError('All stages are unknown')
        stages[0] = not_unknown[0]
        print(idx_unknown)
        idx_unknown = idx_unknown[1:]
        print(idx_unknown)

    # Remove UNKNOWNs by extrapolating the previous stage
    for i in idx_unknown:
        stages[i] = stages[i-1]

    return stages


def read_psd(device_label, psd_data_dir):
    """Reads the pickled PSD info.

    Args:
        device_label (str):  
        psd_data_dir (str): path to the directry containing the PDSs

    Returns:
        [list]: normalized PSDs
    """
    print('Reading PSD...')

    # read the normalized EEG PSDs and the associated normalization factors and means
    pkl_path_eeg = os.path.join(psd_data_dir, f'{device_label}_EEG_PSD.pkl')
    with open(pkl_path_eeg, 'rb') as pkl:
        spec_norm_eeg = pickle.load(pkl)

    # read the normalized EMG PSDs and the associated normalization factors and means
    pkl_path_emg = os.path.join(psd_data_dir, f'{device_label}_EMG_PSD.pkl')
    with open(pkl_path_emg, 'rb') as pkl:
        spec_norm_emg = pickle.load(pkl)

    return spec_norm_eeg, spec_norm_emg


def conv_mat(spec_norm_eeg, spec_norm_emg, epoch_num):
    """make the conventional (not standardized) PSD

    Args:
        spec_norm_eeg (matrix): normalized (standardized) sepctrum matrix of EEG
        spec_norm_emg (matrix): normalized (standardized) sepctrum matrix of EMG
        epoch_num (int): The total number of epochs in the record

    Returns:
        [list]: conventional PSD matrices of EEG and EMG
    """
    bidx_unknown = spec_norm_eeg['bidx_unknown']
    nf = spec_norm_eeg['norm_fac']
    nm = spec_norm_eeg['mean']

    eeg_norm_mat = np.repeat(np.nan, epoch_num*len(nf)
                             ).reshape([epoch_num, len(nf)])
    eeg_norm_mat[~bidx_unknown] = spec_norm_eeg['psd']
    eeg_conv_mat = np.vectorize(
        log_psd_inv)(eeg_norm_mat, nf, nm)

    nf = spec_norm_emg['norm_fac']
    nm = spec_norm_emg['mean']
    emg_norm_mat = np.repeat(np.nan, epoch_num*len(nf)
                             ).reshape([epoch_num, len(nf)])
    emg_norm_mat[~bidx_unknown] = spec_norm_emg['psd']
    emg_conv_mat = np.vectorize(
        log_psd_inv)(emg_norm_mat, nf, nm)

    return eeg_conv_mat, emg_conv_mat


def log_psd_inv(y, normalizing_fac, normalizing_mean):
    """ inverses the spectrum normalization to get the conventional PSD
    """
    return 10**((y / normalizing_fac + normalizing_mean) / 10)


def _initialize_axes(axes, day_len_epoch, epoch_len_sec):
    """Initializes the axes
    """
    hourly_xticks = np.arange(0, day_len_epoch, 60*60/epoch_len_sec)

    for axes_set in axes:
        for axis in axes_set[0:4]:
            axis.spines['bottom'].set_linewidth(AXIS_W)
            axis.spines['top'].set_linewidth(AXIS_W)
            axis.spines['left'].set_linewidth(AXIS_W)
            axis.spines['right'].set_linewidth(AXIS_W)
            axis.set_xticks(hourly_xticks)
            axis.set_xticklabels([])
            axis.tick_params(length=0)
        axes_set[0].set_yticks([0.5, 1.5, 2.5])
        axes_set[0].set_yticklabels(
            ['Wake', 'NREM', 'REM'], fontsize=6, fontfamily='arial')
        axes_set[0].set_xlim(0, day_len_epoch)
        axes_set[0].set_ylim(-0.5, 3.5)
        axes_set[1].set_ylabel(
            r'$\delta$ power''\n''[% of total]', fontsize=6, fontfamily='arial')
        axes_set[1].set_yticks([0, 3, 6])
        axes_set[1].set_yticklabels([0, 3, 6], fontsize=6)
        axes_set[1].set_ylim(0, 8)
        axes_set[1].set_xlim(0, day_len_epoch)
        axes_set[1].grid(dashes=(2, 2), linewidth=LINE_W)
        axes_set[2].set_ylabel(r'$\theta$/$\delta$',
                               fontsize=6, fontfamily='arial')
        axes_set[2].set_yticks([0, 3, 6])
        axes_set[2].set_yticklabels([0, 3, 6], fontsize=6)
        axes_set[2].set_ylim(0, 8)
        axes_set[2].set_xlim(0, day_len_epoch)
        axes_set[2].grid(dashes=(2, 2), linewidth=LINE_W)
        axes_set[3].set_ylabel('EMG\n[AU]', fontsize=6, fontfamily='arial')
        axes_set[3].set_yticks([0.00, 0.05, 0.10])
        axes_set[3].set_yticklabels(['0', '', '1'], fontsize=6)
        axes_set[3].set_ylim(0, 0.15)
        axes_set[3].set_xlim(0, day_len_epoch)
        axes_set[3].grid(dashes=(2, 2), linewidth=LINE_W)
        axes_set[4].axis('off')

    last_axes = axes[-1][-2]
    last_axes.set_xlabel('hours')


def run_main():
    """The main function
    """
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--data_dir", required=True,
                            help="path to the directory of input voltage data")
        parser.add_argument("-r", "--result_dir", required=True,
                            help="path to the directory of the input stage data and plots to be produced")
        parser.add_argument(
            "-s", "--stage_ext", help="the sub-extention of the stage file (default: faster2)")
        parser.add_argument(
            "-o", "--origin_hour", help="The starting time of the hypnogram (hours relative to the recording start, default: 0)")
        parser.add_argument(
            "-e", "--end_hour", help="The ending time of the hypnogram (hours relative to the recording start, default: the ending hour of the recording)")
        parser.add_argument(
            "-i", "--index", help="The zero-start comma separated indices of the mouse in mouse.info.csv to draw the hypnogram (default: draw hypnogram for the all mice)")
        parser.add_argument("-l", "--epoch_len_sec", help="epoch length in second", default=8)

        args = parser.parse_args()
        epoch_len_sec = int(args.epoch_len_sec)

        ## data parameters
        data_dir = os.path.abspath(args.data_dir)
        exp_info_df = stage.read_exp_info(data_dir)
        result_dir = os.path.join(args.result_dir)
        stage_ext = args.stage_ext if args.stage_ext else 'faster2'
        psd_data_dir = os.path.join(result_dir, 'PSD')

        (epoch_num, sample_freq, exp_label, rack_label, start_datetime,
         end_datetime) = stage.interpret_exp_info(exp_info_df, epoch_len_sec)
        mouse_info_df = stage.read_mouse_info(data_dir)

        ## plot parameters
        # The range of the plot in hour
        start_hour = int(args.origin_hour) if args.origin_hour else 0
        end_hour = int(args.end_hour) if args.end_hour else epoch_num * epoch_len_sec / 3600
        length_hour = end_hour - start_hour
        n_days = int(np.ceil(length_hour/24))
        print(
            f'Hypnogram range in hour (Origin:{start_hour}, End:{end_hour}, Length:{length_hour})')
        # The range of the plot in epoch
        start_epoch = int(start_hour * 3600 / epoch_len_sec)
        end_epoch = int(end_hour * 3600 / epoch_len_sec)
        day_len_epoch = int(24 * 60 * 60 / epoch_len_sec)
        length_epoch = (end_epoch - start_epoch)
        print(
            f'Hypnogram range in epoch (Origin:{start_epoch}, End:{end_epoch}, Length:{length_epoch})')
        target_indices = [int(x) for x in args.index.split(',')] if args.index else range(len(mouse_info_df))

        ## analysis parameters
        # assures frequency bins compatible among different sampling frequencies
        n_fft = int(256 * sample_freq/100)
        # same frequency bins given by signal.welch()
        freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)
        bidx_theta_freq = (freq_bins >= 4) & (freq_bins < 10)  # 15 bins
        bidx_delta_freq = (freq_bins < 4)  # 11 bins
        bidx_muscle_freq = (freq_bins > 30)  # 52 bins
        n_muscle_freq = np.sum(bidx_muscle_freq)

        # make output directory
        root_plot_dir = os.path.join(result_dir, 'figure', 'hypnogram')
        os.makedirs(root_plot_dir, exist_ok=True)
        os.makedirs(os.path.join(root_plot_dir, 'pdf'), exist_ok=True)

        # plotting
        dt_now = datetime.now()
        print(f'Started plotting hypnograms: {dt_now}')

        for idx, arow in mouse_info_df.iterrows():
            device_label, mouse_group, mouse_id = [
                x.strip() for x in arow[['Device label', 'Mouse group', 'Mouse ID']]]
            print(
                f'Hypnogram [{idx}]: {exp_label} [{device_label}] {mouse_group} ({mouse_id})')
            if idx not in target_indices:
                print(f'Skipping...')
                continue

            ## regularized stages
            print(f'Reading stage... {device_label}.{stage_ext}.stage.csv')
            stages = et.read_stages(result_dir, device_label, stage_ext)
            stages = remove_unknown_stage(stages)
            stage_idx_dict = {'WAKE': 0, 'NREM': 1, 'REM': 2}
            nstages = np.array([stage_idx_dict[stg]
                                for stg in stages])  # numeric stages

            ## make boxes of stages
            # binary index of stage shifts
            # Note that the first stage is always 'shifted'
            bidx_stage_shift = np.concatenate(
                ([True], nstages[:-1] != nstages[1:]))
            # x positions of boxes
            x_pos = np.where(bidx_stage_shift)[0]
            # y positions of boxes
            y_pos = nstages[bidx_stage_shift]
            # widths of boxes
            widths = np.concatenate(
                (x_pos[1:] - x_pos[:-1], [epoch_num - x_pos[-1]]))
            # Color of boxes
            color_of_boxes = COLOR_LIST[y_pos]

            ## PSD data prep.
            spec_norm_eeg, spec_norm_emg = read_psd(device_label, psd_data_dir)
            eeg_conv_mat, emg_conv_mat = conv_mat(
                spec_norm_eeg, spec_norm_emg, epoch_num)
            # conventional power
            cpower_total = np.apply_along_axis(
                np.sum, 1, eeg_conv_mat)
            cpower_theta = np.apply_along_axis(
                np.sum, 1, eeg_conv_mat[:, bidx_theta_freq])
            cpower_delta = np.apply_along_axis(
                np.sum, 1, eeg_conv_mat[:, bidx_delta_freq])
            cpower_muscle = np.apply_along_axis(
                np.sum, 1, emg_conv_mat[:, bidx_muscle_freq])
            # conventional percentage power
            cppower_theta = 100*cpower_theta/cpower_total
            cppower_delta = 100*cpower_delta/cpower_total
            ratio_theta_delta = cppower_theta/cppower_delta
            # conventional average EMG power
            capower_muscle = cpower_muscle / n_muscle_freq

            # PSD lines
            color_of_lines = COLOR_LIST[nstages]
            lines_delta = np.array([((i, 0), (i, cpower))
                                    for i, cpower in enumerate(cpower_delta)])
            lines_rtd = np.array([((i, 0), (i, cpowerr))
                                  for i, cpowerr in enumerate(ratio_theta_delta)])
            lines_muscle = np.array([((i, 0), (i, capower))
                                     for i, capower in enumerate(capower_muscle)])

            # Drawing
            fig = Figure(facecolor="w")
            # The inch size of A4 paper when there are 5 days
            fig.set_size_inches(8.267717, n_days * 11.69291 / 5)
            # top=0.97 regardless of the number of days (fig size)
            fig.subplots_adjust(left=None, bottom=None,
                                right=None, top=1-0.35/(n_days * 11.6929/5), wspace=0, hspace=0.8) 
            gs = fig.add_gridspec(nrows=13*n_days, ncols=1)

            # prepare n_days blocks of axes (0.stage, 1.delta, 2.delta/theta, 3.muscle, 4.spacper)
            axes = [
                [fig.add_subplot(gs[(0 + 13*i):(3 + 13*i), :], xmargin=0, ymargin=0),
                 fig.add_subplot(gs[(3 + 13*i):(6 + 13*i), :], xmargin=0, ymargin=0),
                 fig.add_subplot(gs[(6 + 13*i):(9 + 13*i), :], xmargin=0, ymargin=0),
                 fig.add_subplot(gs[(9 + 13*i):(12+13*i), :], xmargin=0, ymargin=0),
                 fig.add_subplot(gs[(12+13*i):(13+13*i), :], xmargin=0, ymargin=0)]
                for i in range(n_days)]

            _initialize_axes(axes, day_len_epoch, epoch_len_sec)

            if length_hour >= 24:
                for i, range_start_epoch in enumerate(np.arange(start_epoch, end_epoch, day_len_epoch)):
                    range_end_epoch = min(
                        range_start_epoch + day_len_epoch, end_epoch)
                    # Boxex of stages
                    start_block_idx = max(
                        np.where(x_pos > range_start_epoch)[0][0] - 1, 0)
                    end_block_idx = min(np.where(x_pos < range_end_epoch)[0][-1] + 1, len(x_pos))
                    range_x_pos = x_pos[start_block_idx:end_block_idx]
                    range_y_pos = y_pos[start_block_idx:end_block_idx]
                    range_widths = widths[start_block_idx:end_block_idx]
                    boxes = [Rectangle((x_pos - range_start_epoch, y_pos), wid, 1) for x_pos, y_pos, wid
                             in zip(range_x_pos,
                                    range_y_pos,
                                    range_widths)]

                    patch_collection = PatchCollection(
                        boxes, facecolor=color_of_boxes[start_block_idx:end_block_idx])
                    axes[i][0].add_collection(patch_collection)

                    # line colors
                    range_line_colors = color_of_lines[range_start_epoch:range_end_epoch]

                    # delta power
                    range_line_delta = lines_delta[range_start_epoch:range_end_epoch] - [
                        [range_start_epoch, 0], [range_start_epoch, 0]]
                    line_collection_delta = LineCollection(range_line_delta,
                                                           color=range_line_colors,
                                                           linewidths=LINE_W)
                    axes[i][1].add_collection(line_collection_delta)

                    # theta/delta ratio
                    range_lines_rtd = lines_rtd[range_start_epoch:range_end_epoch] - [
                        [range_start_epoch, 0], [range_start_epoch, 0]]
                    line_collection_rtd = LineCollection(
                        range_lines_rtd, color=range_line_colors, linewidths=LINE_W)
                    axes[i][2].add_collection(line_collection_rtd)

                    # muscle
                    range_lines_muscle = lines_muscle[range_start_epoch:range_end_epoch] - [
                        [range_start_epoch, 0], [range_start_epoch, 0]]
                    line_collection_muscle = LineCollection(
                        range_lines_muscle, color=range_line_colors, linewidths=LINE_W)
                    axes[i][3].add_collection(line_collection_muscle)

                    # x ticks
                    hourly_labels = [str(hour_label + start_hour + (24 * i))
                                     for hour_label in range(24)]
                    axes[i][3].set_xticklabels(
                        hourly_labels, fontsize=6, fontfamily='arial')

            else:
                range_start_epoch = start_epoch
                range_end_epoch = end_epoch
                # Boxex of stages
                start_block_idx = max(
                    np.where(x_pos > range_start_epoch)[0][0] - 1, 0)
                end_block_idx = min(np.where(x_pos < range_end_epoch)[0][-1] + 1, len(x_pos))
                range_x_pos = x_pos[start_block_idx:end_block_idx]
                range_y_pos = y_pos[start_block_idx:end_block_idx]
                range_widths = widths[start_block_idx:end_block_idx]
                boxes = [Rectangle((x_pos - range_start_epoch, y_pos), wid, 1) for x_pos, y_pos, wid
                         in zip(range_x_pos,
                                range_y_pos,
                                range_widths)]

                patch_collection = PatchCollection(
                    boxes, facecolor=color_of_boxes)
                axes[0][0].add_collection(patch_collection)
                axes[0][0].set_xlim(0, length_epoch)

                # line colors
                range_line_colors = color_of_lines[range_start_epoch:range_end_epoch]

                # delta power
                range_line_delta = lines_delta[range_start_epoch:range_end_epoch] - [
                    [range_start_epoch, 0], [range_start_epoch, 0]]
                line_collection_delta = LineCollection(range_line_delta,
                                                       color=range_line_colors,
                                                       linewidths=LINE_W)
                axes[0][1].add_collection(line_collection_delta)
                axes[0][1].set_xlim(0, length_epoch)

                # theta/delta ratio
                range_lines_rtd = lines_rtd[range_start_epoch:range_end_epoch] - [
                    [range_start_epoch, 0], [range_start_epoch, 0]]
                line_collection_rtd = LineCollection(
                    range_lines_rtd, color=range_line_colors, linewidths=LINE_W)
                axes[0][2].add_collection(line_collection_rtd)
                axes[0][2].set_xlim(0, length_epoch)

                # muscle
                range_lines_muscle = lines_muscle[range_start_epoch:range_end_epoch] - [
                    [range_start_epoch, 0], [range_start_epoch, 0]]
                line_collection_muscle = LineCollection(
                    range_lines_muscle, color=range_line_colors, linewidths=LINE_W)
                axes[0][3].add_collection(line_collection_muscle)

                # x ticks
                hourly_labels = [str(hour_label + start_hour)
                                 for hour_label in range(24)]
                axes[0][3].set_xticklabels(
                    hourly_labels, fontsize=6, fontfamily='arial')
                axes[0][3].set_xlim(0, length_epoch)

            # y=0.99 regardless of the number of days (fig size)
            fig.suptitle(
                f'Hypnogram: {exp_label} ({device_label}) [{mouse_group}] {mouse_id} ', 
                fontsize=10, fontfamily='arial', y=1-0.117/(n_days * 11.6929/5)) 
            filename = f"hypnogram_{exp_label}_{mouse_group}_{mouse_id}_{device_label}"
            out_path = os.path.join(root_plot_dir, filename)
            print(f'Saving PNG...')
            fig.savefig(f"{out_path}.png", dpi=FIG_DPI,
                        bbox_inches='tight', pad_inches=2/FIG_DPI)
            out_path = os.path.join(root_plot_dir, 'PDF', filename)
            print(f'Saving PDF...')
            fig.savefig(f"{out_path}.pdf", dpi=FIG_DPI,
                        bbox_inches='tight', pad_inches=2/FIG_DPI)

        elapsed_time = (datetime.now() - dt_now)
        print(
            f'Ended plotting: {datetime.now()},'
            f'ellapsed {elapsed_time.total_seconds()/60} minuites')


run_main()
