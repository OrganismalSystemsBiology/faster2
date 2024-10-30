# -*- coding: utf-8 -*-
""" Useful functions for handling PSD
"""
import os
import copy
import warnings
import textwrap
from logging import getLogger
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import faster2lib.eeg_tools as et
import faster2lib.summary_common as sc
import stage


DOMAIN_NAMES = ['Slow', 'Delta w/o slow', 'Delta', 'Theta']

SPECTROGRAM_FIG_WIDTH = 8
SPECTROGRAM_FIG_HEIGHT = 4
FIG_DPI = 100
COLOR_LIGHT = '#FFD700'  # 'gold'
COLOR_DARK = '#696969'  # 'dimgray'
COLOR_PSD_PEAK = 'red'   # peak frequency
COLOR_PSD_CM = 'purple'  # center of mass
COLOR_PSD_DCM = 'orange' # delta center of mass


LOGGER = getLogger(__name__)

def psd_freq_bins(sample_freq):
    """ assures frequency bins compatibe among different sampling frequencies

    Args:
        sample_freq (int): The sampling frequency

    Returns:
        np.array: An array of frequency bins
    """
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    return freq_bins


def get_bidx_theta_freq(freq_bins):
    """ returns the index of theta frequency bins
        Args:
            freq_bins (np.array): An array of frequency bins
        Returns:
            np.array: The binary index of theta frequency bins for theta waves 
    """
    bidx_theta_freq = (freq_bins >= 4) & (freq_bins < 10)
    return bidx_theta_freq

def get_bidx_delta_freq(freq_bins):
    """ returns the index of delta frequency bins
        Args:
            freq_bins (np.array): An array of frequency bins
        Returns:
            np.array: The binary index of delta frequency bins for delta waves 
    """
    # Notice: This is slightly different from the definition of delta in the stage.py
    # The stage.py defines delta including 0 Hz for simplicity.
    # Here, we exclude 0 Hz from delta for compatibility with the conventional definition.
    bidx_delta_freq = (0 < freq_bins) & (freq_bins < 4)
    return bidx_delta_freq

def get_bidx_delta_wo_slow_freq(freq_bins):
    """ returns thef index of delta frequency bins without slow waves
        Args:
            freq_bins (np.array): An array of frequency bins
        Returns:
            np.array: The binary index of delta frequency bins without slow waves for delta waves
        """
    bidx_delta_wo_slow_freq = (1 <= freq_bins) & (freq_bins < 4)
    return bidx_delta_wo_slow_freq

def get_bidx_slow_freq(freq_bins):
    """ returns the index of slow frequency bins
        Args:
            freq_bins (np.array): An array of frequency bins
        Returns:
            np.array: The binary index of slow frequency bins for slow waves
        """
    bidx_slow_freq = (0 < freq_bins ) & ( freq_bins < 1)
    return bidx_slow_freq


def make_psd_profile(psd_info_list, sample_freq, psd_type='norm', mask=None):
    """makes summary PSD statistics of each mouse:
            psd_mean_df: summary (default: mean) of PSD profiles for each stage for each mice.

    Arguments:
        psd_info_list {[np.array]} -- a list of psd_info given by make_target_psd_info()
        sample_freq {int} -- sampling frequency
        psd_type {str}: 'norm' or 'raw'
        mask {str} -- binary mask to specify epochs to be included. Default is None meaning
                      to include all epochs
    Returns:
        psd_summary_df {pd.DataFrame} --  Experiment label, Mouse group, Mouse ID,
            Device label, Stage, [freq_bins...]

    """

    def _psd_summary_by_bidx(bidx):
        if np.sum(bidx) > 0:
            psd_summary = np.apply_along_axis(np.nanmean, 0, conv_psd[bidx, :])
        else:
            psd_summary = np.full(conv_psd.shape[1], np.nan)

        return psd_summary

    freq_bins = psd_freq_bins(sample_freq)

    psd_summary_df = pd.DataFrame()
    for psd_info in psd_info_list:
        device_label = psd_info['device_label']
        mouse_group = psd_info['mouse_group']
        mouse_id = psd_info['mouse_id']
        exp_label = psd_info['exp_label']
        conv_psd = psd_info[psd_type]

        # Default mask
        if mask is None:
            mask = np.full(len(psd_info['bidx_target']), True)

        bidx_rem_target = psd_info['bidx_rem'] & psd_info['bidx_target'] & mask
        bidx_nrem_target = psd_info['bidx_nrem'] & psd_info['bidx_target'] & mask
        bidx_wake_target = psd_info['bidx_wake'] & psd_info['bidx_target'] & mask

        psd_summary_rem = _psd_summary_by_bidx(bidx_rem_target)
        psd_summary_nrem = _psd_summary_by_bidx(bidx_nrem_target)
        psd_summary_wake = _psd_summary_by_bidx(bidx_wake_target)

        psd_summary_df = pd.concat([psd_summary_df, 
            pd.DataFrame([
            [exp_label, mouse_group, mouse_id, device_label,
             'REM', np.sum(bidx_rem_target)] + psd_summary_rem.tolist()])], ignore_index=True)
        psd_summary_df = pd.concat([psd_summary_df, 
            pd.DataFrame([
            [exp_label, mouse_group, mouse_id, device_label,
             'NREM', np.sum(bidx_nrem_target)] + psd_summary_nrem.tolist()])], ignore_index=True)
        psd_summary_df = pd.concat([psd_summary_df, 
            pd.DataFrame([
            [exp_label, mouse_group, mouse_id, device_label,
             'Wake', np.sum(bidx_wake_target)] + psd_summary_wake.tolist()])], ignore_index=True)

    freq_columns = [f'f@{x}' for x in freq_bins.tolist()]
    column_names = ['Experiment label', 'Mouse group',
                    'Mouse ID', 'Device label', 'Stage', 'epoch #'] + freq_columns
    psd_summary_df.columns = column_names

    return psd_summary_df


def make_target_psd_info(mouse_info_df, epoch_range, epoch_len_sec, sample_freq,
                         stage_ext):
    """makes PSD information sets for subsequent statistical analysis for each mouse:
    Arguments:
        mouse_info_df {pd.DataFram} -- a dataframe given by mouse_info_collected()
        sample_freq {int} -- sampling frequency
        epoch_range {slice} -- a range of target epochs
        stage_ext {str} -- a file sub-extention (e.g. 'faster2' for *.faster2.csv)

    Returns:
        psd_info_list [dict] --  A list of dict:
            'exp_label', 'mouse_group':mouse group, 'mouse_id':mouse id,
            'device_label', 'stage_call', 'bidx_rem', 'bidx_nrem','bidx_wake',
            'bidx_target', 'bidx_unknown', 'norm', 'raw'
    NOTE:
        stage.py also makes PSD. But, it is different from PSD of summary.py
        because stage.py dicards PSD of unknown epochs while here the length of
        conv_psd is always equal to len(stage).
    """

    # target PSDs are from known-stage's, within-the-epoch-range, and good epochs
    psd_info_list = []
    for i, row in mouse_info_df.iterrows():
        device_label = row['Device label'].strip()
        stats_report = row['Stats report'].strip().upper()
        mouse_group = row['Mouse group'].strip()
        mouse_id = row['Mouse ID'].strip()
        exp_label = row['Experiment label'].strip()
        faster_dir = row['FASTER_DIR']
        start_datetime = row['exp_start_datetime']

        if stats_report == 'NO':
            LOGGER.info('[%d] Skipping: %s %s', i+1, faster_dir, device_label)
            continue
        LOGGER.info('[%d] Reading stage of: %s %s', i+1, faster_dir, device_label)


        # read stage of the mouse
        try:
            stage_call, nan_eeg, outlier_eeg = et.read_stages_with_eeg_diagnosis(os.path.join(
                faster_dir, 'result'), device_label, stage_ext)
        except IndexError:
            # Manually annotated stage files may not have diagnostic info
            LOGGER.info('NA and outlier information is not available in the stage file')
            stage_call = et.read_stages(os.path.join(
                faster_dir, 'result'), device_label, stage_ext)
            nan_eeg = np.repeat(0, len(stage_call))
            outlier_eeg = np.repeat(0, len(stage_call))
        epoch_num = len(stage_call)

        # Read the voltage (EMG is also necessary for marking unknown epochs)
        # pylint: disable=unused-variable
        (eeg_vm_org, emg_vm_org, not_yet_pickled) = et.read_voltage_matrices(
            os.path.join(faster_dir, 'data'), device_label, sample_freq, epoch_len_sec,
            epoch_num, start_datetime)

        LOGGER.info('Preprocessing and calculating PSD')
        # recover nans in the voltage if possible
        np.apply_along_axis(et.patch_nan, 1, eeg_vm_org)
        np.apply_along_axis(et.patch_nan, 1, emg_vm_org)

        # exclude unrecoverable epochs as unknown (note: this process involves the EMG)
        bidx_unknown = np.apply_along_axis(np.any, 1, np.isnan(
            eeg_vm_org)) | np.apply_along_axis(np.any, 1, np.isnan(emg_vm_org))

        # Break at the error:
        # the unknown epoch is not recoverable (even a manual annotator may want)
        if not np.all(stage_call[bidx_unknown] == 'UNKNOWN'):
            LOGGER.info('[Error] Unknown stages are inconsistent between the PSD and stage files.'\
                      'Note unknown stages are not recoverable by manual annotation. '\
                      'Check the consistency between the PSD and the stage files.')
            idx = list(np.where(bidx_unknown)[
                0][stage_call[bidx_unknown] != 'UNKNOWN'])
            LOGGER.info('... in stage file at %d', idx)
            break

        # good PSD should have the nan- and outlier-ratios of less than 10%
        bidx_good_psd = (nan_eeg < 0.10) & (outlier_eeg < 0.10)

        # bidx_target: bidx for the good epochs in the selected range
        bidx_selected = np.repeat(False, epoch_num)
        bidx_selected[epoch_range] = True
        bidx_target = bidx_selected & bidx_good_psd & ~bidx_unknown

        LOGGER.info('    Target epoch range: %d-%d (%d epochs out of %d epochs)\n'\
                    '    Unknown epochs in the range: %d (%.3f %%)\n'\
                    '    Outlier or NA epochs in the range: %d (%.3f %%)',
                    epoch_range.start, epoch_range.stop,
                    epoch_range.stop - epoch_range.start, epoch_num,
                    np.sum(bidx_unknown & bidx_selected), 100 *
                    np.sum(bidx_unknown & bidx_selected)/np.sum(bidx_selected),
                    np.sum(~bidx_good_psd & bidx_selected),
                    100*np.sum(~bidx_good_psd & bidx_selected)/np.sum(bidx_selected))


        bidx_rem = (stage_call == 'REM') & bidx_target
        bidx_nrem = (stage_call == 'NREM') & bidx_target
        bidx_wake = (stage_call == 'WAKE') & bidx_target

        # normalize the EEG voltage matrix with balanced number of Wake & NREM epochs
        epoch_num_nrem = np.sum(bidx_nrem)
        epoch_num_wake = np.sum(bidx_wake)
        epoch_num_balanced = np.min([epoch_num_nrem, epoch_num_wake])
        idx_nrem = np.where(bidx_nrem)[0]
        idx_wake = np.where(bidx_wake)[0]
        np.random.shuffle(idx_nrem)
        np.random.shuffle(idx_wake)
        idx_nrem = idx_nrem[:epoch_num_balanced]
        idx_wake = idx_wake[:epoch_num_balanced]
        volts = eeg_vm_org[np.hstack([idx_nrem, idx_wake]), :]
        eeg_vm_norm = (eeg_vm_org - np.nanmean(volts))/(np.nanstd(volts))
        LOGGER.info('    Number of epochs sampled for normalization: %d each from '\
                    'NREM (%d epochs) and Wake (%d epochs)',
                    epoch_num_balanced, epoch_num_nrem, epoch_num_wake)
        # calculate EEG's PSD
        # assures frequency bins are compatible among different sampling frequencies
        n_fft = int(256 * sample_freq/100)
        # Note: unknown epoch's conv_psd is nan. This is different from PSD pickled by stage.py.
        conv_psd = np.apply_along_axis(
            lambda y, nfft=n_fft: stage.psd(y, nfft, sample_freq), 1, eeg_vm_norm)
        # PSD without normalization
        conv_psd_raw = np.apply_along_axis(
            lambda y, nfft=n_fft: stage.psd(y, nfft, sample_freq), 1, eeg_vm_org)

        psd_info_list.append({'exp_label': exp_label,
                              'mouse_group': mouse_group,
                              'mouse_id': mouse_id,
                              'device_label': device_label,
                              'stage_call': stage_call,
                              'bidx_rem': bidx_rem,
                              'bidx_nrem': bidx_nrem,
                              'bidx_wake': bidx_wake,
                              'bidx_unknown': bidx_unknown,
                              'bidx_target': bidx_target,
                              'freq_bins': psd_freq_bins(sample_freq),
                              'norm': conv_psd,
                              'raw': conv_psd_raw})

    return psd_info_list


def make_psd_domain(psd_profiles_df, summary_func=np.mean):
    """ makes PSD power averaged within frequency domains of each stage for each mice

    Arguments:
        psd_profiles_df {pd.DataFrame} -- a dataframe given by make_psd_profile()
        summary_func {function} -- a function for summarizing PSD

    Returns:
        [pd.DataFrame] -- Experiment label, Mouse group, Mouse ID, 
            Device label, Stage, Slow, Delta w/o slow, Delta, Theta
    """
    # get freq_bins from column names
    freq_bin_columns = psd_profiles_df.columns[6:].tolist()
    freq_bins = np.array([float(x.strip().split('@')[1])
                          for x in freq_bin_columns])

    # frequency domains
    bidx_theta_freq = get_bidx_theta_freq(freq_bins)
    bidx_delta_freq = get_bidx_delta_freq(freq_bins)
    bidx_delta_wo_slow_freq = get_bidx_delta_wo_slow_freq(freq_bins) # delta without slow
    bidx_slow_freq = get_bidx_slow_freq(freq_bins)

    # make psd_domain_df
    row_list = []
    for _, r in psd_profiles_df.iterrows():
        infos = r[:6]
        powers = r[6:]
        powers_slow = powers[bidx_slow_freq]
        powers_delta_wo_slow = powers[bidx_delta_wo_slow_freq]
        powers_delta = powers[bidx_delta_freq]
        powers_theta = powers[bidx_theta_freq]

        slow_p = summary_func(powers_slow)
        delta_wo_slow_p = summary_func(powers_delta_wo_slow)
        delta_p = summary_func(powers_delta)
        theta_p = summary_func(powers_theta)
        domain_powers = pd.Series(
            [slow_p, delta_wo_slow_p, delta_p, theta_p], index=DOMAIN_NAMES)

        row = pd.concat([infos, domain_powers])
        row_list.append(row)

    psd_domain_df = pd.DataFrame(row_list)
    return psd_domain_df


def make_psd_stats(psd_domain_df, summary_func):
    """ makes a table of statistical tests for each frequency domains between groups 

    Arguments:
        psd_domain_df {pd.DataFrame} -- a dataframe given by make_psd_domain()
        summary_func {function} -- a function for summarizing PSD

    Returns:
        [pd.DataFrame] -- # mouse_group, stage_type, wave_type, num, 
                            mean, SD, pvalue, star, method
    """
    def _domain_powers_by_group(psd_domain_df, group):
        bidx_group = (psd_domain_df['Mouse group'] == group)
        bidx_rem = (psd_domain_df['Stage'] == 'REM')
        bidx_nrem = (psd_domain_df['Stage'] == 'NREM')
        bidx_wake = (psd_domain_df['Stage'] == 'Wake')

        domain_powers_rem = psd_domain_df.loc[bidx_group &
                                              bidx_rem][DOMAIN_NAMES]
        domain_powers_nrem = psd_domain_df.loc[bidx_group &
                                               bidx_nrem][DOMAIN_NAMES]
        domain_powers_wake = psd_domain_df.loc[bidx_group &
                                               bidx_wake][DOMAIN_NAMES]

        return [domain_powers_rem, domain_powers_nrem, domain_powers_wake]

    if summary_func is np.sum:
        summary_str = 'Sum'
    else:
        summary_str = 'Mean'

    psd_stats_df = pd.DataFrame()
    # mouse_group_set
    mouse_group_list = psd_domain_df['Mouse group'].tolist()
    # unique elements with preseved order
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)

    # control
    group_c = mouse_group_set[0]  # index=0 should be always control group

    # There are 3 powers_domains_[stages] where [stages] are [REM, NREM, Wake].
    # Each powers_domains_[stage] contains 4 Series of domain powers:
    # [slow x mice, delta wo slow x mice, delta x mice, theta x mice]
    # Therefore, the following loop results in 12 rows.
    stage_names = ['REM', 'NREM', 'Wake']
    powers_domains_stages_c = _domain_powers_by_group(psd_domain_df, group_c)

    rows = []
    for stage_name, powers_domains in zip(stage_names, powers_domains_stages_c):
        for domain_name in DOMAIN_NAMES:
            powers = powers_domains[domain_name]
            num = np.sum(~np.isnan(powers)) # conunt effective N
            rows.append([group_c, stage_name, domain_name, num,
                         np.mean(powers),  np.std(powers), np.nan, None, None])

    psd_stats_df = pd.concat([psd_stats_df, pd.DataFrame(rows)], ignore_index=True)

    # treatment
    for group_t in mouse_group_set[1:]:
        rows = []
        powers_domains_stages_t = _domain_powers_by_group(
            psd_domain_df, group_t)
        for stage_name, powers_domains_c, powers_domains_t in zip(stage_names, powers_domains_stages_c, powers_domains_stages_t):
            for domain_name in DOMAIN_NAMES:
                powers_c = powers_domains_c[domain_name]
                powers_t = powers_domains_t[domain_name]
                test = sc.test_two_sample(powers_c, powers_t)
                num = np.sum(~np.isnan(powers_t)) # conunt effective N
                rows.append([group_t, stage_name, domain_name, num,
                             np.nanmean(powers_t),  np.nanstd(powers_t), test['p_value'], test['stars'], test['method']])

        psd_stats_df = pd.concat([psd_stats_df, pd.DataFrame(rows)], ignore_index=True)

    psd_stats_df.columns = ['Mouse group', 'Stage type',
                            'Wake type', 'N', summary_str, 'SD', 'Pvalue', 'Stars', 'Method']

    return psd_stats_df


def make_psd_timeseries_df(psd_info_list, epoch_range, bidx_freq, stage_bidx_key=None, 
                           psd_type='norm', scaling_type='none', transform_type='linear'):
    """make timeseries of PSD with a specified stage and freq domain

    Args:
        psd_info_list (list of psd_info): The list of object given by make_target_psd_info()
        epoch_range (slice): The epoch range of interest
        stage_bidx_key (str): The key of dict in the psd_info for a stage (e.g. 'bidx_nrem')
        psd_type (str): 'norm' or 'raw'

    Returns:
        [pd.dataframe]: The timeseries of PSD
    """
    psd_timeseries_df = pd.DataFrame()
    for psd_info in psd_info_list:
        bidx_target = psd_info['bidx_target']
        if stage_bidx_key:
            bidx_stage = psd_info[stage_bidx_key]
            bidx_targeted_stage = bidx_target & bidx_stage
        else:
            bidx_targeted_stage = bidx_target

        conv_psd = psd_info[psd_type]
        psd_delta_timeseries = np.repeat(np.nan, epoch_range.stop - epoch_range.start)
        if scaling_type == 'AUC' and transform_type == 'linear':
            # Use summation for delta power only if scaling_type is AUC and transform_type is linear
            psd_delta_timeseries[bidx_targeted_stage[epoch_range]] = np.apply_along_axis(np.nansum, 1, conv_psd[bidx_targeted_stage, :][:,bidx_freq])
        else:
            # Use mean for delta power for other cases
            psd_delta_timeseries[bidx_targeted_stage[epoch_range]] = np.apply_along_axis(np.nanmean, 1, conv_psd[bidx_targeted_stage, :][:,bidx_freq])
        psd_timeseries_df = pd.concat([psd_timeseries_df,
            pd.DataFrame([[psd_info['exp_label'], psd_info['mouse_group'], psd_info['mouse_id'], psd_info['device_label']] + psd_delta_timeseries.tolist()])], ignore_index=True)

    epoch_columns = [f'epoch{x+1}' for x in np.arange(epoch_range.start, epoch_range.stop)]
    column_names = ['Experiment label', 'Mouse group', 'Mouse ID', 'Device label'] + epoch_columns
    psd_timeseries_df.columns = column_names

    return psd_timeseries_df


def write_psd_stats(psd_profiles_df, output_dir, opt_label='', summary_func=np.mean):
    """ writes three PSD tables:
        1. psd_profile.csv: mean PSD profile of each stage for each mice
        2. psd_freq_domain_table.csv: PSD power averaged within frequency domains of each stage for each mice 
        3. psd_stats_table.csv: statistical tests for each frequency domains between groups 
    """

    psd_domain_df = make_psd_domain(psd_profiles_df, summary_func)
    psd_stats_df = make_psd_stats(psd_domain_df, summary_func)

    # write tabels
    psd_profiles_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile.csv'), index=False)
    psd_domain_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile_freq_domain_table.csv'), index=False)
    psd_stats_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile_stats_table.csv'), index=False)


def draw_PSDs_individual(psd_profiles_df, sample_freq, y_label, output_dir, 
                         psd_type, scaling_type, transform_type, opt_label=''):
    freq_bins = psd_freq_bins(sample_freq)

    # mouse_set
    mouse_list = psd_profiles_df['Mouse ID'].tolist()
    # unique elements with preseved order
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

    pre_proc = f'{psd_type}_{scaling_type}_{transform_type}'
    # draw individual PSDs
    for m in mouse_set:
        fig = Figure(figsize=(16, 4))
        fig.subplots_adjust(wspace=0.27)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel(f'REM\n{y_label}')
        ax2.set_xlabel('freq. [Hz]')
        ax2.set_ylabel(f'NREM\n{y_label}')
        ax3.set_xlabel('freq. [Hz]')
        ax3.set_ylabel(f'Wake\n{y_label}')

        x = freq_bins
        df = psd_profiles_df.loc[psd_profiles_df['Mouse ID'] == m]
        y = df.loc[df['Stage'] == 'REM'].iloc[0].values[6:]
        ax1.plot(x, y, color=stage.COLOR_REM)

        y = df.loc[df['Stage'] == 'NREM'].iloc[0].values[6:]
        ax2.plot(x, y, color=stage.COLOR_NREM)

        y = df.loc[df['Stage'] == 'Wake'].iloc[0].values[6:]
        ax3.plot(x, y, color=stage.COLOR_WAKE)

        mouse_tag_list = [str(x) for x in df.iloc[0, 0:4]]
        fig.suptitle(
            f'Powerspectrum density: {"  ".join(mouse_tag_list)}\n'
            f'Preprocessed with: (Voltage,Scaling,Transformation) = ({psd_type}, {scaling_type}, {transform_type})',
            y=1.05)
        filename = f'PSD_{pre_proc}_{opt_label}profile_I_{"_".join(mouse_tag_list)}'
        sc.savefig(output_dir, filename, fig)


def draw_PSDs_group(psd_profiles_df, sample_freq, y_label, output_dir, 
                    psd_type, scaling_type, transform_type, opt_label=''):
    freq_bins = psd_freq_bins(sample_freq)

    # mouse_group_set
    mouse_group_list = psd_profiles_df['Mouse group'].tolist()
    # unique elements with preseved order
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)
    pre_proc = f'{psd_type}_{scaling_type}_{transform_type}'
    
    # draw gropued PSD
    # _c of Control (assuming index = 0 is a control mouse)
    df = psd_profiles_df[psd_profiles_df['Mouse group'] == mouse_group_set[0]]

    psd_mean_mat_rem_c = df[df['Stage'] == 'REM'].iloc[:, 6:].values
    psd_mean_mat_nrem_c = df[df['Stage'] == 'NREM'].iloc[:, 6:].values
    psd_mean_mat_wake_c = df[df['Stage'] == 'Wake'].iloc[:, 6:].values
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

    x = freq_bins
    if len(mouse_group_set)>1:
        for g_idx in range(1, len(mouse_group_set)):
            fig = Figure(figsize=(16, 4))
            fig.subplots_adjust(wspace=0.27)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.set_xlabel('freq. [Hz]')
            ax1.set_ylabel(f'REM\n{y_label}')
            ax2.set_xlabel('freq. [Hz]')
            ax2.set_ylabel(f'NREM\n{y_label}')
            ax3.set_xlabel('freq. [Hz]')
            ax3.set_ylabel(f'Wake\n{y_label}')

            # _t of Treatment
            df = psd_profiles_df[psd_profiles_df['Mouse group']
                                == mouse_group_set[g_idx]]
            psd_mean_mat_rem_t = df[df['Stage'] == 'REM'].iloc[:, 6:].values
            psd_mean_mat_nrem_t = df[df['Stage'] == 'NREM'].iloc[:, 6:].values
            psd_mean_mat_wake_t = df[df['Stage'] == 'Wake'].iloc[:, 6:].values
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
                f'Powerspectrum density: {mouse_group_set[0]} (n={num_c}) v.s. {mouse_group_set[g_idx]} (n={num_t})\n'
                f'Preprocessed with: (Voltage,Scaling,Transformation) = ({psd_type}, {scaling_type}, {transform_type})',
                y=1.05)
            filename = f'PSD_{pre_proc}_{opt_label}profile_G_{mouse_group_set[0]}_vs_{mouse_group_set[g_idx]}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0
        fig = Figure(figsize=(16, 4))
        fig.subplots_adjust(wspace=0.27)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel(f'REM\n{y_label}')
        ax2.set_xlabel('freq. [Hz]')
        ax2.set_ylabel(f'NREM\n{y_label}')
        ax3.set_xlabel('freq. [Hz]')
        ax3.set_ylabel(f'Wake\n{y_label}')

        # _t of Treatment
        df = psd_profiles_df[psd_profiles_df['Mouse group']
                            == mouse_group_set[g_idx]]
        psd_mean_mat_rem_t = df[df['Stage'] == 'REM'].iloc[:, 6:].values
        psd_mean_mat_nrem_t = df[df['Stage'] == 'NREM'].iloc[:, 6:].values
        psd_mean_mat_wake_t = df[df['Stage'] == 'Wake'].iloc[:, 6:].values
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

        y = psd_mean_rem_t
        y_sem = psd_sem_rem_t
        ax1.plot(x, y, color=stage.COLOR_REM)
        ax1.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_REM, alpha=0.3)

        y = psd_mean_nrem_t
        y_sem = psd_sem_nrem_t
        ax2.plot(x, y, color=stage.COLOR_NREM)
        ax2.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

        y = psd_mean_wake_t
        y_sem = psd_sem_wake_t
        ax3.plot(x, y, color=stage.COLOR_WAKE)
        ax3.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_WAKE, alpha=0.3)

        fig.suptitle(
            f'Powerspectrum density: {mouse_group_set[g_idx]} (n={num_t})\n'
            f'Preprocessed with: (Voltage,Scaling,Transformation) = ({psd_type}, {scaling_type}, {transform_type})',
            y=1.05)
        filename = f'PSD_{pre_proc}_{opt_label}profile_G_{mouse_group_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)


def draw_psd_domain_power_timeseries_individual(psd_domain_power_timeseries_df, epoch_len_sec, y_label, output_dir, 
                                                psd_type, scaling_type, transform_type, domain, opt_label=''):
    mouse_groups = psd_domain_power_timeseries_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]
    pre_proc = f'{psd_type}_{scaling_type}_{transform_type}'

    hourly_ts_list = []
    for _, ts in psd_domain_power_timeseries_df.iloc[:,4:].iterrows():
        ts_mat = ts.to_numpy(dtype=np.float64).reshape(-1, int(3600/epoch_len_sec))
        # The rows with all nan needs to be avoided in np.nanmean
        idx_all_nan = np.where([np.all(np.isnan(x)) for x in ts_mat])
        ts_mat[idx_all_nan, :] = 0
        hourly_ts = np.apply_along_axis(np.nanmean, 1, ts_mat)
        hourly_ts[idx_all_nan] = np.nan # matplotlib can handle np.nan
        hourly_ts_list.append(hourly_ts)

    hourly_ts_mat = np.array(hourly_ts_list)

    # this is just for deciding y_max
    domain_power_timeseries_stats_list=[]
    for bidx in bidx_group_list:
        hourly_ts_mat_group = hourly_ts_mat[bidx]
        idx_all_nan = np.where([np.all(np.isnan(r)) for r in hourly_ts_mat_group.T])
        hourly_ts_mat_group[:, idx_all_nan] = 0 # this is for np.nanmean and np.nanstd
        domain_power_timeseries_mean = np.apply_along_axis(
            np.nanmean, 0, hourly_ts_mat_group)
        domain_power_timeseries_sd = np.apply_along_axis(
            np.nanstd, 0, hourly_ts_mat_group)
        domain_power_timeseries_mean[idx_all_nan] = np.nan
        domain_power_timeseries_sd[idx_all_nan] = np.nan
        domain_power_timeseries_stats_list.append(
            np.array([domain_power_timeseries_mean, domain_power_timeseries_sd]))
    y_vals = np.array([ts_stats[0] for ts_stats in domain_power_timeseries_stats_list])
    y_max = np.nanmax(y_vals)
    y_min = np.nanmin(y_vals)

    x_max = ts_mat.shape[0]
    x = np.arange(x_max)
    for i, profile in enumerate(hourly_ts_list):
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)
        sc.set_common_features_domain_power_timeseries(ax1, x_max, y_min, y_max)

        ax1.set_ylabel(y_label)
        ax1.set_xlabel('Time (hours)')

        ax1.plot(x, profile, color=stage.COLOR_NREM)

        fig.suptitle(
            f'Power timeseries: {"  ".join(psd_domain_power_timeseries_df.iloc[i,0:4].values)}\n'
            f'Preprocessed with: (Voltage,Scaling,Transformation) = ({psd_type}, {scaling_type}, {transform_type})')

        filename = f'power-timeseries_{pre_proc}_{domain}_{opt_label}I_{"_".join(psd_domain_power_timeseries_df.iloc[i,0:4].values)}'
        sc.savefig(output_dir, filename, fig)


def draw_psd_domain_power_timeseries_grouped(psd_domain_power_timeseries_df, epoch_len_sec, y_label, output_dir, 
                                             psd_type, scaling_type, transform_type, domain, opt_label=''):
    """Draws power timeseries grouped by mouse groups
    """
    mouse_groups = psd_domain_power_timeseries_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]  
    pre_proc = f'{psd_type}_{scaling_type}_{transform_type}'

    hourly_ts_list = []
    for _, ts in psd_domain_power_timeseries_df.iloc[:,4:].iterrows():
        ts_mat = ts.to_numpy().reshape(-1, int(3600/epoch_len_sec))
        # The rows with all nan needs to be avoided in np.nanmean
        idx_all_nan = np.where([np.all(np.isnan(x)) for x in ts_mat])
        ts_mat[idx_all_nan, :] = 0
        hourly_ts = np.apply_along_axis(np.nanmean, 1, ts_mat)
        hourly_ts[idx_all_nan] = np.nan # matplotlib can handle np.nan
        hourly_ts_list.append(hourly_ts)
    hourly_ts_mat = np.array(hourly_ts_list)

    domain_power_timeseries_stats_list=[]
    for bidx in bidx_group_list:
        hourly_ts_mat_group = hourly_ts_mat[bidx]
        idx_all_nan = np.where([np.all(np.isnan(r)) for r in hourly_ts_mat_group.T])
        hourly_ts_mat_group[:, idx_all_nan] = 0 # this is for np.nanmean and np.nanstd
        domain_power_timeseries_mean = np.apply_along_axis(
            np.nanmean, 0, hourly_ts_mat_group)
        domain_power_timeseries_sd = np.apply_along_axis(
            np.nanstd, 0, hourly_ts_mat_group)
        domain_power_timeseries_mean[idx_all_nan] = np.nan
        domain_power_timeseries_sd[idx_all_nan] = np.nan
        domain_power_timeseries_stats_list.append(
            np.array([domain_power_timeseries_mean, domain_power_timeseries_sd]))

    # pylint: disable=E1136  # pylint/issues/3139
    x_max = hourly_ts_mat.shape[1]
    y_vals = np.array([ts_stats[0] for ts_stats in domain_power_timeseries_stats_list])
    y_max = np.nanmax(y_vals)
    y_min = np.nanmin(y_vals)

    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
            num = np.sum(bidx_group_list[g_idx])
            fig = Figure(figsize=(13, 6))
            ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)

            sc.set_common_features_domain_power_timeseries(ax1, x_max, y_min, y_max)

            # Control (always the first group)
            num_c = np.sum(bidx_group_list[0])
            y = domain_power_timeseries_stats_list[0][0, :]
            y_sem = domain_power_timeseries_stats_list[0][1, :]/np.sqrt(num_c)
            ax1.plot(x, y, color='grey')
            ax1.fill_between(x, y - y_sem,
                            y + y_sem, color='grey', alpha=0.3)
            ax1.set_ylabel(y_label)
            ax1.set_xlabel('Time (hours)')

            # Treatment
            num = np.sum(bidx_group_list[g_idx])
            y = domain_power_timeseries_stats_list[g_idx][0, :]
            y_sem = domain_power_timeseries_stats_list[g_idx][1, :]/np.sqrt(num)
            ax1.plot(x, y, color=stage.COLOR_NREM)
            ax1.fill_between(x, y - y_sem,
                            y + y_sem, color=stage.COLOR_NREM, alpha=0.3)

            fig.suptitle(
                f'Power timeseries: {mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})\n'
                f'Preprocessed with: (Voltage,Scaling,Transformation) = ({psd_type}, {scaling_type}, {transform_type})'
            )
            filename = f'power-timeseries_{pre_proc}_{domain}_{opt_label}G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0
        num = np.sum(bidx_group_list[g_idx])
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)

        sc.set_common_features_domain_power_timeseries(ax1, x_max, y_min, y_max)

        y = domain_power_timeseries_stats_list[g_idx][0, :]
        y_sem = domain_power_timeseries_stats_list[g_idx][1, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax1.set_ylabel(y_label)
        ax1.set_xlabel('Time (hours)')

        fig.suptitle(f'Power timeseries: {mouse_groups_set[g_idx]} (n={num})\n'
                     f'Preprocessed with: (Voltage,Scaling,Transformation) = ({psd_type}, {scaling_type}, {transform_type})'
        )
        filename = f'power-timeseries_{pre_proc}_{domain}_{opt_label}G_{mouse_groups_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)


def _center_of_mass_ts(psd_ts, freq_bins, bidx_freq_bin_mask=None):
    ''' Calculate the center of mass time series.
    Args:
        psd_ts (np.ndarray): The PSD-profile time series. (timepoints x freq_bins)
        freq_bins (np.ndarray): The frequency bins.
        bidx_freq_bin_mask (np.ndarray): The mask for the frequency bins; true to INCLUDE.
    Returns:
        center_of_mass_ts (np.ndarray): The center of mass time series.
    '''
    psd_ts_tmp = copy.deepcopy(psd_ts) # stash the original data
    if bidx_freq_bin_mask is not None:
        psd_ts_tmp[:, ~bidx_freq_bin_mask] = 0
    center_of_mass_ts = np.array([np.dot(freq_bins, (x.T)/np.nansum(x)) if np.nansum(x) != 0 else np.nan for x in psd_ts_tmp])
    return center_of_mass_ts


def _get_psd_profile_ts(psd_info, psd_type, epoch_range):
    ''' Get the PSD-profile time series from the PSD information.
    Args:
        psd_info (dict): The PSD information.
        psd_type (str): The type of PSD data to extract. 'raw' or 'norm'.
        epoch_range (slice): The range of epochs to extract the PSD-profile time series.
    Returns:
        psd_prof_ts (np.ndarray): The PSD-profile time series. (freq bins x time points)    
    '''
    psd_mat = psd_info[psd_type]
    bidx_nrem = psd_info['bidx_nrem']
    bidx_target = psd_info['bidx_target']
    psd_mat[~(bidx_nrem & bidx_target), :] = np.nan

    return psd_mat[epoch_range, :]


def _circ_bin_psd_profile_ts(psd_prof_binned_ts, bin_len_min):
    ''' Circadian binned PSD-profile time series.
    Args:
        psd_prof_binned_ts (np.ndarray): The PSD-profile binned time series. (time bins x freq bins)
        bin_len_min (int): The length of the bin in minutes.
    Returns:
        psd_prof_binned_ts_24h (np.ndarray): The 24-hour binned PSD-profile time series. (24 bins x freq bins)
    '''
    with warnings.catch_warnings():
        # There may be all nan segments. It is safe to ignore the warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psd_avg_prof_binned_ts_24h = np.nanmean(psd_prof_binned_ts.reshape(-1, int(24/(bin_len_min/60)), psd_prof_binned_ts.shape[1]), axis=0)
        
    return psd_avg_prof_binned_ts_24h


def _bin_psd_profile_ts(psd_prof_ts, epoch_num_in_bin):
    ''' Bin the PSD-profile time series.
    Args:
        psd_prof_ts (np.ndarray): The PSD-profile time series. (time points x freq bins)
        epoch_num_in_bin (int): The number of epochs in a bin.
    Returns:
        psd_prof_binned_ts (np.ndarray): The PSD-profile binned time series. (time bins x freq bins)
    '''
    with warnings.catch_warnings():
        # There may be all nan segments. It is safe to ignore the warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psd_prof_binned_ts = np.nanmean(psd_prof_ts.reshape(-1, epoch_num_in_bin, psd_prof_ts.shape[1]), axis=1)
    return psd_prof_binned_ts


def _draw_psd_peak_circ_heatmap(mat_for_heatmap, peak_ts_avg, peak_ts_sem, cm_ts_avg, cm_ts_sem, dcm_ts_avg, dcm_ts_sem, freq_bins, unit_label, global_v_max=None):
    """ Draws PSD peak frequency circadian heatmap
    Args:
        mat_for_heatmap (np.ndarray): The matrix for heatmap (timepoints x freq_bins)
        peak_ts_avg (np.ndarray): The average of peak frequency
        peak_ts_sem (np.ndarray): The standard error of peak frequency
        cm_ts_avg (np.ndarray): The average of center of mass
        cm_ts_sem (np.ndarray): The standard error of center of mass
        dcm_ts_avg (np.ndarray): The average of delta-domain center of mass
        dcm_ts_sem (np.ndarray): The standard error of delta-domain center of mass
        freq_bins (np.ndarray): The frequency bins
        unit_label (str): The unit of y-axis
    Returns:
        [matplotlib.figure.Figure]: The figure
    """
    # plot parameters
    freq_max_idx = 30
    datalen_24h = mat_for_heatmap.shape[0]

    # initialize the figure
    fig = Figure(figsize=(SPECTROGRAM_FIG_WIDTH,
                          SPECTROGRAM_FIG_HEIGHT), dpi=stage.FIG_DPI, facecolor='w')
    ax = fig.add_subplot(111)

    # prepare subcidial parameters
    tp = np.arange(datalen_24h)
    fb = freq_bins[:freq_max_idx] # frequency bins
    if global_v_max is None:
        v_max = np.percentile(mat_for_heatmap, 99) # use the global 99% percentile as the maximum
    else:
        v_max = global_v_max
    y_max = freq_bins[freq_max_idx]
    ax.set_ylim(-1, y_max)

    # light bar
    light_bar_base = Rectangle(
        xy=[0, -1], width=24, height=0.8, fill=True, color=COLOR_DARK)
    ax.add_patch(light_bar_base)

    light_bar_light = Rectangle(
        xy=[12, -1], width=12, height=0.8, fill=True, color=COLOR_LIGHT)
    ax.add_patch(light_bar_light)

    # draw
    pcm = ax.pcolormesh(tp + 0.5, fb, mat_for_heatmap[:,:freq_max_idx].T, vmax=v_max, cmap='viridis')
    ax.plot(tp + 0.5, peak_ts_avg, color=COLOR_PSD_PEAK, label='Peak frequency')
    ax.fill_between(tp + 0.5, peak_ts_avg - peak_ts_sem, peak_ts_avg + peak_ts_sem, color=COLOR_PSD_PEAK, linewidth=0, alpha=0.3)
    ax.plot(tp + 0.5, cm_ts_avg, color='C1', label='Center of mass')
    ax.fill_between(tp + 0.5, cm_ts_avg - cm_ts_sem, cm_ts_avg + cm_ts_sem, color='C1', linewidth=0, alpha=0.3)
    ax.plot(tp + 0.5, dcm_ts_avg, color=COLOR_PSD_CM, label='Delta-domain center of mass')
    ax.fill_between(tp + 0.5, dcm_ts_avg - dcm_ts_sem, dcm_ts_avg + dcm_ts_sem, color=COLOR_PSD_CM, linewidth=0, alpha=0.3)

    # labels
    ax.set_yticks(freq_bins[:freq_max_idx:5], [f'{x:.2f}' for x in freq_bins[:freq_max_idx:5]])
    ax.set_xticks(np.linspace(0, datalen_24h, 7), np.linspace(0, 24, 7).astype(int))
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(pcm, ax=ax, label=f'Power [{unit_label}]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=5)
    
    return fig


def make_peak_freq_ts_mat(psd_info_list, epoch_range, psd_type, epoch_len_sec):
    ''' Make a collection of timeseries matricies of peak frequency and its relatives.
    Args:
        psd_info_list (list): The list of PSD information.
        epoch_range (slice): The range of epochs to extract the PSD-profile time series.
        psd_type (str): The type of PSD data to extract. 'raw' or 'norm'.
        epoch_len_sec (int): The length of the epoch in seconds.
    Returns:
        tuple of two arrays: The binned time series of the PSD profile and the 24-hour binned average PSD-profile.
        
    '''
    bin_len_min = 60 # bin width is 60 minutes
    wrk_psd_info_list = copy.deepcopy(psd_info_list) # stash the original psd_info_list 
    epoch_num_in_bin = int(bin_len_min * 60 / epoch_len_sec)
    
    # calculate the timeseries of PSD profile (freq bins x time points)
    psd_prof_ts_list = []
    psd_prof_binned_ts_list = []
    psd_prof_binned_ts_24h_list = []
    for psd_info in wrk_psd_info_list:
        # Obtain the PSD-profile time series during NREM for each individual (freq bins x time points)
        psd_prof_ts = _get_psd_profile_ts(psd_info, psd_type, epoch_range)
        # Obtain the binned time series of the PSD profile during NREM for each individual (freq bins x time bins)
        psd_prof_binned_ts = _bin_psd_profile_ts(psd_prof_ts, epoch_num_in_bin)
        # Obtain the 24-hour binned average PSD-profile from the binned time series of the PSD profile
        psd_avg_prof_binned_ts_24h = _circ_bin_psd_profile_ts(psd_prof_binned_ts, bin_len_min)
        
        # Store the results
        psd_prof_ts_list.append(psd_prof_ts)
        psd_prof_binned_ts_list.append(psd_prof_binned_ts)
        psd_prof_binned_ts_24h_list.append(psd_avg_prof_binned_ts_24h)

    
    return np.array(psd_prof_binned_ts_list), np.array(psd_prof_binned_ts_24h_list)


def peak_freq_stats_ts_dict(psd_prof_ts_mat, freq_bins):
    ''' Calculate the timeseries matrix of the peak frequency and its relatives.
    Args:
        psd_prof_ts_list (np.ndarray): The matrix of PSD-profile time series.
        freq_bins (np.ndarray): The frequency bins.
        bidx_freq_bin_mask (np.ndarray): The mask for the frequency bins; true to INCLUDE.
    Returns:
        dict of peak_freq, center of mass, center of mass in the delta domain
    '''
    # center of mass
    cm_ts_mat = np.array([_center_of_mass_ts(x, freq_bins) for x in psd_prof_ts_mat])

    # delta-center of mass
    bidx_delta =  get_bidx_delta_freq(freq_bins)
    delta_cm_ts_mat = np.array([_center_of_mass_ts(x, freq_bins, bidx_delta) for x in psd_prof_ts_mat])

    # peak frequency
    idx_peak_mat = np.array([x.T.argmax(axis=0) for x in psd_prof_ts_mat])
    bidx_empty = np.all(np.isnan(psd_prof_ts_mat), axis=2) # to behave like 'nanargmax'
    peak_ts_mat = np.array([freq_bins[idx] for idx in idx_peak_mat])
    peak_ts_mat[bidx_empty] = np.nan

    return {'cm_ts_mat':cm_ts_mat, 'delta_cm_ts_mat':delta_cm_ts_mat, 'peak_ts_mat':peak_ts_mat}


def calc_ts_stats_from_ts_mat(ts_mat):
    ''' Calculate the timeseries of average, standard deviation, and the number of non-nan elements from the time series matrix.
    Args:
        ts_mat (np.ndarray): The time series matrix.
    Returns:
        avg (np.ndarray): The average.
        std (np.ndarray): The standard deviation.
        n (np.ndarray): The number of non-nan elements.
    '''
    with warnings.catch_warnings():
        # There may be all nan segments. It is safe to ignore the warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg = np.nanmean(ts_mat, axis=0)
        std = np.nanstd(ts_mat, axis=0, ddof=1)
    n = np.sum(~np.isnan(ts_mat), axis=0)
    return avg, std, n


def draw_psd_peak_circ_heatmap_indiviudal(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, unit_label, output_dir):
    """ Draws PSD peak frequency circadian heatmap for individual mouse
    Args:
        psd_info_list (list of psd_info): The list of object given by make_target_psd_info()
        psd_type (str): 'norm' or 'raw'
        scaling_type (str): 'none' or 'AUC' or 'TDD'
        transform_type (str): 'linear' or 'log'
        unit_label (str): The unit of y-axis
        output_dir (str): The output directory  
    """

    if transform_type == 'log':
        # log transformation is not supported
        return

    freq_bins = psd_info_list[0]['freq_bins']

    psd_prof_binned_ts_mat, psd_prof_binned_ts_24h_mat = make_peak_freq_ts_mat(psd_info_list, epoch_range, psd_type, epoch_len_sec)
    pf_stats_ts_dict     = peak_freq_stats_ts_dict(psd_prof_binned_ts_mat,     freq_bins)

    datalen_24h = psd_prof_binned_ts_24h_mat.shape[1]
    global_v_max = np.nanpercentile(psd_prof_binned_ts_24h_mat, 99)
    # loop for indiviudal mouse
    for psd_info_idx, psd_info in enumerate(psd_info_list):
        mat_for_heatmap = psd_prof_binned_ts_24h_mat[psd_info_idx, :, :]
        mouse_group = psd_info['mouse_group']
        mouse_id = psd_info['mouse_id']
        device_label = psd_info['device_label']
        exp_label = psd_info['exp_label']

        # data preparation
        peak_ts_vec = pf_stats_ts_dict['peak_ts_mat'][psd_info_idx]
        cm_ts_vec = pf_stats_ts_dict['cm_ts_mat'][psd_info_idx]
        dcm_ts_vec = pf_stats_ts_dict['delta_cm_ts_mat'][psd_info_idx]
        peak_ts_avg, peak_ts_std, peak_ts_n = calc_ts_stats_from_ts_mat(peak_ts_vec.reshape(-1, datalen_24h))
        peak_ts_sem = peak_ts_std / np.sqrt(peak_ts_n)
        cm_ts_avg, cm_ts_std, cm_ts_n = calc_ts_stats_from_ts_mat(cm_ts_vec.reshape(-1, datalen_24h))
        cm_ts_sem = cm_ts_std / np.sqrt(cm_ts_n)
        dcm_ts_avg, dcm_ts_std, dcm_ts_n = calc_ts_stats_from_ts_mat(dcm_ts_vec.reshape(-1, datalen_24h))
        dcm_ts_sem = dcm_ts_std / np.sqrt(dcm_ts_n)

        fig = _draw_psd_peak_circ_heatmap(mat_for_heatmap, peak_ts_avg, peak_ts_sem, cm_ts_avg, cm_ts_sem, dcm_ts_avg, dcm_ts_sem, freq_bins, unit_label, global_v_max)
        
        fig.get_axes()[0].set_title(f'PSD-peak-circ_heatmap: {exp_label} {mouse_group} {mouse_id} {device_label}\n'
                f'Processed with: (voltage dist, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})', fontsize=10)
        filename = f'PSD-peak-circ_heatmap_{psd_type}_{scaling_type}_{transform_type}_I_{exp_label}_{mouse_group}_{mouse_id}_{device_label}'
        sc.savefig(output_dir, filename, fig)  


def draw_psd_peak_circ_heatmap_grouped(psd_info_list,  epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, unit_label, output_dir):
    """ Draws PSD peak frequency circadian heatmap for grouped mouse
    Args:
        psd_info_list (list of psd_info): The list of object given by make_target_psd_info()
        epoch_range (slice): The range of epochs to extract the PSD-profile time series.
        epoch_len_sec (int): The length of the epoch in seconds.
        psd_type (str): 'norm' or 'raw'
        scaling_type (str): 'none' or 'AUC' or 'TDD'
        transform_type (str): 'linear' or 'log'
        unit_label (str): The unit of y-axis
        output_dir (str): The output directory
    """

    if transform_type == 'log':
        # log transformation is not supported
        return

    freq_bins = psd_info_list[0]['freq_bins']
    
    mouse_groups = np.array([psd_info['mouse_group'] for psd_info in psd_info_list])
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    
    # prep data
    _, psd_prof_binned_ts_24h_mat = make_peak_freq_ts_mat(psd_info_list, epoch_range, psd_type, epoch_len_sec)
    pf_stats_ts_dict_24h = peak_freq_stats_ts_dict(psd_prof_binned_ts_24h_mat, freq_bins)

    for grp in mouse_groups_set:
        bidx_grp = (mouse_groups == grp)
        mat_for_heatmap = np.nanmean(psd_prof_binned_ts_24h_mat[bidx_grp], axis=0)

        # data preparation
        peak_ts_mat = pf_stats_ts_dict_24h['peak_ts_mat'][bidx_grp]
        cm_ts_mat   = pf_stats_ts_dict_24h['cm_ts_mat'][bidx_grp]
        dcm_ts_mat  = pf_stats_ts_dict_24h['delta_cm_ts_mat'][bidx_grp]
        peak_ts_avg, peak_ts_std, peak_ts_n = calc_ts_stats_from_ts_mat(peak_ts_mat)
        peak_ts_sem = peak_ts_std / np.sqrt(peak_ts_n)
        cm_ts_avg, cm_ts_std, cm_ts_n = calc_ts_stats_from_ts_mat(cm_ts_mat)
        cm_ts_sem = cm_ts_std / np.sqrt(cm_ts_n)
        dcm_ts_avg, dcm_ts_std, dcm_ts_n = calc_ts_stats_from_ts_mat(dcm_ts_mat)
        dcm_ts_sem = dcm_ts_std / np.sqrt(dcm_ts_n)

        fig = _draw_psd_peak_circ_heatmap(mat_for_heatmap, peak_ts_avg, peak_ts_sem, cm_ts_avg, cm_ts_sem, dcm_ts_avg, dcm_ts_sem, freq_bins, unit_label)

        fig.get_axes()[0].set_title(f'PSD-peak-circ_heatmap: {grp} (n={np.sum(bidx_grp)})\n'
                f'Processed with: (voltage dist, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})', fontsize=10)
        filename = f'PSD-peak-circ_heatmap_{psd_type}_{scaling_type}_{transform_type}_G_{grp}'
        sc.savefig(output_dir, filename, fig)   


def draw_psd_peak_circ_lineplot(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, output_dir):
    """ Draws PSD peak frequency circadian lineplot for grouped mouse
    Args:
        psd_info_list (list of psd_info): The list of object given by make_target_psd_info()
        epoch_range (slice): The range of epochs to extract the PSD-profile time series.
        epoch_len_sec (int): The length of the epoch in seconds.
        psd_type (str): 'norm' or 'raw'
        scaling_type (str): 'none' or 'AUC' or 'TDD'
        transform_type (str): 'linear' or 'log'
        output_dir (str): The output directory
    """
    if transform_type == 'log':
        # log transformation is not supported
        return

    def _light_bar(ax, light_bar_ratio):
        y_min, y_max = ax.get_ylim()
        bar_height = (y_max - y_min) * light_bar_ratio

        # light bar
        light_bar_base = Rectangle(
            xy=[0, -bar_height], width=24, height=bar_height, fill=True, color=COLOR_DARK)
        ax.add_patch(light_bar_base) 

        light_bar_light = Rectangle(
            xy=[12, -bar_height], width=12, height=bar_height, fill=True, color=COLOR_LIGHT)
        ax.add_patch(light_bar_light)
    
    freq_max_idx = 30
    freq_bins = psd_info_list[0]['freq_bins']

    mouse_groups = np.array([psd_info['mouse_group'] for psd_info in psd_info_list])
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order

    psd_prof_binned_ts_mat, psd_prof_binned_ts_24h_mat = make_peak_freq_ts_mat(psd_info_list, epoch_range, psd_type, epoch_len_sec)
    pf_stats_ts_dict_24h = peak_freq_stats_ts_dict(psd_prof_binned_ts_24h_mat, freq_bins)
    datalen_24h = psd_prof_binned_ts_24h_mat.shape[1]

    grp_ctrl = mouse_groups_set[0]
    bidx_grp_ctrl = (mouse_groups == mouse_groups_set[0])

    # data preparation
    peak_ts_ctrl_mat = pf_stats_ts_dict_24h['peak_ts_mat'][bidx_grp_ctrl]
    cm_ts_ctrl_mat   = pf_stats_ts_dict_24h['cm_ts_mat'][bidx_grp_ctrl]
    dcm_ts_ctrl_mat  = pf_stats_ts_dict_24h['delta_cm_ts_mat'][bidx_grp_ctrl]

    peak_ts_ctrl_avg, peak_ts_ctrl_std, peak_ts_ctrl_n = calc_ts_stats_from_ts_mat(peak_ts_ctrl_mat)
    peak_ts_ctrl_sem = peak_ts_ctrl_std / np.sqrt(peak_ts_ctrl_n)
    costest_peak_ctrl = sc.costest(peak_ts_ctrl_avg, peak_ts_ctrl_sem, len(peak_ts_ctrl_avg), len(peak_ts_ctrl_avg))

    cm_ts_ctrl_avg, cm_ts_ctrl_std, cm_ts_ctrl_n = calc_ts_stats_from_ts_mat(cm_ts_ctrl_mat)
    cm_ts_ctrl_sem = cm_ts_ctrl_std / np.sqrt(cm_ts_ctrl_n)
    costest_cm_ctrl = sc.costest(cm_ts_ctrl_avg, cm_ts_ctrl_sem, len(cm_ts_ctrl_avg), len(cm_ts_ctrl_avg))

    dcm_ts_ctrl_avg, dcm_ts_ctrl_std, dcm_ts_ctrl_n = calc_ts_stats_from_ts_mat(dcm_ts_ctrl_mat)
    dcm_ts_ctrl_sem = dcm_ts_ctrl_std / np.sqrt(dcm_ts_ctrl_n)
    costest_dcm_ctrl = sc.costest(dcm_ts_ctrl_avg, dcm_ts_ctrl_sem, len(dcm_ts_ctrl_avg), len(dcm_ts_ctrl_avg))

    psd_peak_costest_list = [] 
    psd_peak_costest_list.append([grp_ctrl, 'psd_peak', len(peak_ts_ctrl_avg), costest_peak_ctrl[1], costest_peak_ctrl[3]])
    psd_peak_costest_list.append([grp_ctrl, 'psd_cm', len(cm_ts_ctrl_avg), costest_cm_ctrl[1], costest_cm_ctrl[3]])
    psd_peak_costest_list.append([grp_ctrl, 'psd_dcm', len(dcm_ts_ctrl_avg), costest_dcm_ctrl[1], costest_dcm_ctrl[3]])

    # initialize the figure
    fig = Figure(figsize=(13,4), dpi=FIG_DPI, facecolor='w')

    axes = fig.subplots(1, 3, sharex=True, sharey=False)
    y_max = 4.5
    light_bar_ratio = 0.05
    axes[0].set_ylim(-y_max*light_bar_ratio, y_max)
    axes[0].set_ylabel('Peak frequency (Hz)')

    axes[1].set_ylim(-y_max*light_bar_ratio, y_max)
    axes[1].set_ylabel('Delta-CM frequency (Hz)')

    y_max = freq_bins[freq_max_idx]
    axes[2].set_ylim(-y_max*light_bar_ratio, freq_bins[freq_max_idx])
    axes[2].set_ylabel('CM frequency (Hz)')

    axes[0].set_xlim(0, datalen_24h)
    axes[0].set_xticks(np.linspace(0, datalen_24h, 7), np.linspace(0, 24, 7).astype(int))
    axes[0].set_xlabel('Time (hour)')

    _light_bar(axes[0], light_bar_ratio)
    _light_bar(axes[1], light_bar_ratio)
    _light_bar(axes[2], light_bar_ratio)

    tp = np.arange(datalen_24h)
    axes[0].plot(tp, peak_ts_ctrl_avg, color='gray')
    axes[0].fill_between(tp, peak_ts_ctrl_avg - peak_ts_ctrl_sem, peak_ts_ctrl_avg + peak_ts_ctrl_sem, color='gray', linewidth=0, alpha=0.3)
    axes[0].set_title('NREM Peak frequency')

    axes[1].plot(tp, dcm_ts_ctrl_avg, color='gray')
    axes[1].fill_between(tp, dcm_ts_ctrl_avg - dcm_ts_ctrl_sem, dcm_ts_ctrl_avg + dcm_ts_ctrl_sem, color='gray', linewidth=0, alpha=0.3)
    axes[1].set_title('NREM Delta-domain center of mass')

    axes[2].plot(tp, cm_ts_ctrl_avg, color='gray')
    axes[2].fill_between(tp, cm_ts_ctrl_avg - cm_ts_ctrl_sem, cm_ts_ctrl_avg + cm_ts_ctrl_sem, color='gray', linewidth=0, alpha=0.3)
    axes[2].set_title('NREM Center of mass')

    if len(mouse_groups_set) > 1:
        # loop for multiple groups
        for grp_test in mouse_groups_set[1:]:
            bidx_grp_test = (mouse_groups == grp_test)

            # data preparation
            peak_ts_test_mat = pf_stats_ts_dict_24h['peak_ts_mat'][bidx_grp_test]
            cm_ts_test_mat   = pf_stats_ts_dict_24h['cm_ts_mat'][bidx_grp_test]
            dcm_ts_test_mat  = pf_stats_ts_dict_24h['delta_cm_ts_mat'][bidx_grp_test]

            peak_ts_test_avg, peak_ts_test_std, peak_ts_test_n = calc_ts_stats_from_ts_mat(peak_ts_test_mat)
            peak_ts_test_sem = peak_ts_test_std / np.sqrt(peak_ts_test_n)
            costest_peak_test = sc.costest(peak_ts_test_avg, peak_ts_test_sem, len(peak_ts_test_avg), len(peak_ts_test_avg))

            cm_ts_test_avg, cm_ts_test_std, cm_ts_test_n = calc_ts_stats_from_ts_mat(cm_ts_test_mat)
            cm_ts_test_sem = cm_ts_test_std / np.sqrt(cm_ts_test_n)
            costest_cm_test = sc.costest(cm_ts_test_avg, cm_ts_test_sem, len(cm_ts_test_avg), len(cm_ts_test_avg))

            dcm_ts_test_avg, dcm_ts_test_std, dcm_ts_test_n = calc_ts_stats_from_ts_mat(dcm_ts_test_mat)
            dcm_ts_test_sem = dcm_ts_test_std / np.sqrt(dcm_ts_test_n)
            costest_dcm_test = sc.costest(dcm_ts_test_avg, dcm_ts_test_sem, len(dcm_ts_test_avg), len(dcm_ts_test_avg))

            psd_peak_costest_list.append([grp_test, 'psd_peak', len(peak_ts_test_avg), costest_peak_test[1], costest_peak_test[3]])
            psd_peak_costest_list.append([grp_test, 'psd_cm', len(cm_ts_test_avg), costest_cm_test[1], costest_cm_test[3]])
            psd_peak_costest_list.append([grp_test, 'psd_dcm', len(dcm_ts_test_avg), costest_dcm_test[1], costest_dcm_test[3]])

            axes[0].plot(tp, peak_ts_test_avg, color=COLOR_PSD_PEAK)
            axes[0].fill_between(tp, peak_ts_test_avg - peak_ts_test_sem, peak_ts_test_avg + peak_ts_test_sem, color=COLOR_PSD_PEAK, linewidth=0, alpha=0.3)

            axes[1].plot(tp, dcm_ts_test_avg, color=COLOR_PSD_CM)
            axes[1].fill_between(tp, dcm_ts_test_avg - dcm_ts_test_sem, dcm_ts_test_avg + dcm_ts_test_sem, color=COLOR_PSD_CM, linewidth=0, alpha=0.3)

            axes[2].plot(tp, cm_ts_test_avg, color=COLOR_PSD_DCM)
            axes[2].fill_between(tp, cm_ts_test_avg - cm_ts_test_sem, cm_ts_test_avg + cm_ts_test_sem, color='C1', linewidth=0, alpha=0.3)
            fig.suptitle(f'PSD-peak-circ_lineplots: {grp_ctrl} (n={np.sum(bidx_grp_ctrl)}) vs {grp_test} (n={np.sum(bidx_grp_test)})\n'
                        f'Processed with: (voltage dist, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})', y=1.05)
            filename = f'PSD-peak-circ_lineplots_{psd_type}_{scaling_type}_{transform_type}_{grp_ctrl}_vs_{grp_test}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        fig.suptitle(f'PSD-peak-circ_lineplots: {grp_ctrl} (n={np.sum(bidx_grp_ctrl)})\n'
                    f'Processed with: (voltage dist, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})', y=1.05)
        filename = f'PSD-peak-circ_lineplots_{psd_type}_{scaling_type}_{transform_type}_{grp_ctrl}'
        sc.savefig(output_dir, filename, fig)

    psd_peak_costest_df = pd.DataFrame(psd_peak_costest_list, columns=['group', 'PSD-peak type', 'n', 'phase', 'Pvalue'])
    psd_peak_costest_df.to_csv(os.path.join(output_dir, f'PSD-peak-circ_lineplots_costest_{psd_type}_{scaling_type}_{transform_type}.csv'), index=False)
    return fig


def draw_psd_peak_barplot(psd_info_list, epoch_range, epoch_len_sec, psd_type, scaling_type, transform_type, output_dir):
    """ Draws PSD peak frequency circadian barplot for grouped mouse
    Args:
        psd_info_list (list of psd_info): The list of object given by make_target_psd_info()
        epoch_range (slice): The range of epochs to extract the PSD-profile time series.
        epoch_len_sec (int): The length of the epoch in seconds.
        psd_type (str): 'norm' or 'raw'
        scaling_type (str): 'none' or 'AUC' or 'TDD'
        transform_type (str): 'linear' or 'log'
        output_dir (str): The output directory
    """
    if transform_type == 'log':
        # log transformation is not supported
        return

    freq_max_idx = 30
    freq_bins = psd_info_list[0]['freq_bins']

    mouse_groups = np.array([psd_info['mouse_group'] for psd_info in psd_info_list])
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    
    psd_prof_binned_ts_mat, psd_prof_binned_ts_24h_mat = make_peak_freq_ts_mat(psd_info_list, epoch_range, psd_type, epoch_len_sec)
    pf_stats_ts_dict_24h = peak_freq_stats_ts_dict(psd_prof_binned_ts_24h_mat, freq_bins)

    grp_ctrl = mouse_groups_set[0]
    bidx_grp_ctrl = (mouse_groups == mouse_groups_set[0])

    # data preparation
    peak_ts_ctrl_mat = pf_stats_ts_dict_24h['peak_ts_mat'][bidx_grp_ctrl]
    cm_ts_ctrl_mat   = pf_stats_ts_dict_24h['cm_ts_mat'][bidx_grp_ctrl]
    dcm_ts_ctrl_mat  = pf_stats_ts_dict_24h['delta_cm_ts_mat'][bidx_grp_ctrl]

    peak_ctrl_avg_vec = np.nanmean(peak_ts_ctrl_mat, axis=1)
    peak_ctrl_avg = np.nanmean(peak_ctrl_avg_vec)
    peak_ctrl_std = np.nanstd(peak_ctrl_avg_vec, ddof=1)
    peak_ctrl_sem = peak_ctrl_std/np.sum(~np.isnan(peak_ctrl_avg_vec))

    cm_ctrl_avg_vec = np.nanmean(cm_ts_ctrl_mat, axis=1)
    cm_ctrl_avg = np.nanmean(cm_ctrl_avg_vec)
    cm_ctrl_std = np.nanstd(cm_ctrl_avg_vec, ddof=1)
    cm_ctrl_sem = cm_ctrl_std/np.sum(~np.isnan(cm_ctrl_avg_vec))

    dcm_ctrl_avg_vec = np.nanmean(dcm_ts_ctrl_mat, axis=1)
    dcm_ctrl_avg = np.nanmean(dcm_ctrl_avg_vec)
    dcm_ctrl_std = np.nanstd(dcm_ctrl_avg_vec, ddof=1)
    dcm_ctrl_sem = dcm_ctrl_std/np.sum(~np.isnan(dcm_ctrl_avg_vec))

    psd_peak_stats_list = []
    psd_peak_stats_list.append([grp_ctrl, 'psd_peak', np.sum(~np.isnan(peak_ctrl_avg_vec)), peak_ctrl_avg, peak_ctrl_std, np.nan, np.nan, np.nan])
    psd_peak_stats_list.append([grp_ctrl, 'psd_cm', np.sum(~np.isnan(cm_ctrl_avg_vec)), cm_ctrl_avg, cm_ctrl_std, np.nan, np.nan, np.nan])
    psd_peak_stats_list.append([grp_ctrl, 'psd_dcm', np.sum(~np.isnan(dcm_ctrl_avg_vec)), dcm_ctrl_avg, dcm_ctrl_std, np.nan, np.nan, np.nan])
    # initialize the figure
    fig = Figure(figsize=(10,4), dpi=FIG_DPI, facecolor='w')
    fig.subplots_adjust(wspace=0.5)

    axes = fig.subplots(1, 3, sharex=True, sharey=False)
    w = 0.8  # bar width
    xtick_str_list = ['\n'.join(textwrap.wrap(mouse_groups_set[g_idx], 8))
                    for g_idx in range(len(mouse_groups_set))]
    x_pos = np.arange(len(mouse_groups_set))

    y_max = 4.5
    axes[0].set_ylim(0, y_max)
    axes[0].set_ylabel('Peak frequency (Hz)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(xtick_str_list)

    axes[1].set_ylim(0, y_max)
    axes[1].set_ylabel('Delta-CM frequency (Hz)')

    y_max = freq_bins[freq_max_idx]
    axes[2].set_ylim(0, freq_bins[freq_max_idx])
    axes[2].set_ylabel('CM frequency (Hz)')

    axes[0].bar(0, peak_ctrl_avg, yerr=peak_ctrl_sem, align='center',
            width=w, capsize=6, color='gray', alpha=0.6)
    sc.scatter_datapoints(axes[0], w, 0, peak_ctrl_avg_vec)
    axes[0].set_title('NREM Peak frequency')

    axes[1].bar(0, dcm_ctrl_avg, yerr=dcm_ctrl_sem, align='center',
            width=w, capsize=6, color='gray', alpha=0.6)
    sc.scatter_datapoints(axes[1], w, 0, dcm_ctrl_avg_vec)
    axes[1].set_title('NREM Delta-domain center of mass')

    axes[2].bar(0, cm_ctrl_avg, yerr=cm_ctrl_sem, align='center',
            width=w, capsize=6, color='gray', alpha=0.6)
    sc.scatter_datapoints(axes[2], w, 0, cm_ctrl_avg_vec)
    axes[2].set_title('NREM Center of mass')

    if len(mouse_groups_set) > 1:
        for grp_test in mouse_groups_set[1:]:
            bidx_grp_test = (mouse_groups == grp_test)

            # data preparation
            peak_ts_test_mat = pf_stats_ts_dict_24h['peak_ts_mat'][bidx_grp_test]
            cm_ts_test_mat   = pf_stats_ts_dict_24h['cm_ts_mat'][bidx_grp_test]
            dcm_ts_test_mat  = pf_stats_ts_dict_24h['delta_cm_ts_mat'][bidx_grp_test]

            peak_test_avg_vec = np.nanmean(peak_ts_test_mat, axis=1)
            peak_test_avg = np.nanmean(peak_test_avg_vec)
            peak_test_std = np.nanstd(peak_test_avg_vec, ddof=1)
            peak_test_sem = peak_test_std/np.sum(~np.isnan(peak_test_avg_vec))

            cm_test_avg_vec = np.nanmean(cm_ts_test_mat, axis=1)
            cm_test_avg = np.nanmean(cm_test_avg_vec)
            cm_test_std = np.nanstd(cm_test_avg_vec, ddof=1)
            cm_test_sem = np.nanstd(cm_test_avg_vec, ddof=1)/np.sum(~np.isnan(cm_test_avg_vec))

            dcm_test_avg_vec = np.nanmean(dcm_ts_test_mat, axis=1)
            dcm_test_avg = np.nanmean(dcm_test_avg_vec)
            dcm_test_std = np.nanstd(dcm_test_avg_vec, ddof=1)
            dcm_test_sem = dcm_test_std/np.sum(~np.isnan(dcm_test_avg_vec))

            t_peak = sc.test_two_sample(peak_ctrl_avg_vec,  peak_test_avg_vec)  # test for peak frequency
            t_cm = sc.test_two_sample(cm_ctrl_avg_vec,  cm_test_avg_vec)  # test for center of mass
            t_dcm = sc.test_two_sample(dcm_ctrl_avg_vec,  dcm_test_avg_vec)  # test for delta-domain center of mass

            psd_peak_stats_list.append([grp_test, 'psd_peak', np.sum(~np.isnan(peak_test_avg_vec)), peak_test_avg, peak_test_std, 
                                        t_peak['p_value'], t_peak['stars'], t_peak['method']])
            psd_peak_stats_list.append([grp_test, 'psd_cm', np.sum(~np.isnan(cm_test_avg_vec)), cm_test_avg, cm_test_std, 
                                        t_cm['p_value'], t_cm['stars'], t_cm['method']])
            psd_peak_stats_list.append([grp_test, 'psd_dcm', np.sum(~np.isnan(dcm_test_avg_vec)), dcm_test_avg, dcm_test_std, 
                                        t_dcm['p_value'], t_dcm['stars'], t_dcm['method']])

            axes[0].bar(1, peak_test_avg, yerr=peak_test_sem, align='center',
                    width=w, capsize=6, color=COLOR_PSD_PEAK, alpha=0.6)
            sc.scatter_datapoints(axes[0], w, 1, peak_test_avg_vec)

            axes[1].bar(1, dcm_test_avg, yerr=dcm_test_sem, align='center',
                    width=w, capsize=6, color=COLOR_PSD_CM, alpha=0.6)
            sc.scatter_datapoints(axes[1], w, 1, dcm_test_avg_vec)

            axes[2].bar(1, cm_test_avg, yerr=cm_test_sem, align='center',
                    width=w, capsize=6, color=COLOR_PSD_DCM, alpha=0.6)
            sc.scatter_datapoints(axes[2], w, 1, cm_test_avg_vec)

            fig.suptitle(f'PSD-peak-barcharts: {grp_ctrl} (n={np.sum(bidx_grp_ctrl)}) vs {grp_test} (n={np.sum(bidx_grp_test)})\n'
                            f'Processed with: (voltage dist, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})', y=1.05)

            filename = f'PSD-peak-barcharts_{psd_type}_{scaling_type}_{transform_type}_{grp_ctrl}_vs_{grp_test}'
            sc.savefig(output_dir, filename, fig)
    else:
        fig.suptitle(f'PSD-peak-barcharts: {grp_ctrl} (n={np.sum(bidx_grp_ctrl)})\n'
                        f'Processed with: (voltage dist, PSD scaling, PSD transformation) = ({psd_type}, {scaling_type}, {transform_type})', y=1.05)

        filename = f'PSD-peak-barcharts_{psd_type}_{scaling_type}_{transform_type}_{grp_ctrl}'
        sc.savefig(output_dir, filename, fig)

    psd_peak_stats_df = pd.DataFrame(psd_peak_stats_list)
    psd_peak_stats_df.columns=['group', 'PSD peak type', 'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']
    psd_peak_stats_df.to_csv(os.path.join(output_dir, f'PSD-peak-barcharts_stats_{psd_type}_{scaling_type}_{transform_type}.csv'), index=False)