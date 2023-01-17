# -*- coding: utf-8 -*-
""" Useful functions for handling PSD
"""
from logging import getLogger
from matplotlib.figure import Figure
import os
import numpy as np
import pandas as pd
import faster2lib.eeg_tools as et
import faster2lib.summary_common as sc
import stage

DOMAIN_NAMES = ['Slow', 'Delta w/o slow', 'Delta', 'Theta']

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


def make_psd_profile(psd_info_list, sample_freq, psd_type='norm', mask=None):
    """makes summary PSD statics of each mouse:
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
    """makes PSD information sets for subsequent static analysis for each mouse:
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
        (eeg_vm_org, emg_vm_org, not_yet_pickled) = stage.read_voltage_matrices(
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
        # assures frequency bins compatibe among different sampleling frequencies
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
    bidx_theta_freq = (freq_bins >= 4) & (freq_bins < 10)  # 15 bins
    bidx_delta_freq = (freq_bins < 4)  # 11 bins
    bidx_delta_wo_slow_freq = (1 <= freq_bins) & (
        freq_bins < 4)  # 8 bins (delta without slow)
    bidx_slow_freq = (freq_bins < 1)  # 3 bins

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
                            'Wake type', 'N', 'Mean', 'SD', 'Pvalue', 'Stars', 'Method']

    return psd_stats_df


def make_psd_timeseries_df(psd_info_list, epoch_range, bidx_freq, stage_bidx_key=None, psd_type='norm'):
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
    psd_stats_df = make_psd_stats(psd_domain_df)

    # write tabels
    psd_profiles_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile.csv'), index=False)
    psd_domain_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile_freq_domain_table.csv'), index=False)
    psd_stats_df.to_csv(os.path.join(
        output_dir, f'PSD_{opt_label}profile_stats_table.csv'), index=False)


def draw_PSDs_individual(psd_profiles_df, sample_freq, y_label, output_dir, opt_label=''):
    freq_bins = psd_freq_bins(sample_freq)

    # mouse_set
    mouse_list = psd_profiles_df['Mouse ID'].tolist()
    # unique elements with preseved order
    mouse_set = sorted(set(mouse_list), key=mouse_list.index)

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
            f'Powerspectrum density: {"  ".join(mouse_tag_list)}')
        filename = f'PSD_{opt_label}profile_I_{"_".join(mouse_tag_list)}'
        sc.savefig(output_dir, filename, fig)


def draw_PSDs_group(psd_profiles_df, sample_freq, y_label, output_dir, opt_label=''):
    freq_bins = psd_freq_bins(sample_freq)

    # mouse_group_set
    mouse_group_list = psd_profiles_df['Mouse group'].tolist()
    # unique elements with preseved order
    mouse_group_set = sorted(set(mouse_group_list), key=mouse_group_list.index)

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
                f'Powerspectrum density: {mouse_group_set[0]} (n={num_c}) v.s. {mouse_group_set[g_idx]} (n={num_t})')
            filename = f'PSD_{opt_label}profile_G_{mouse_group_set[0]}_vs_{mouse_group_set[g_idx]}'
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
            f'Powerspectrum density: {mouse_group_set[g_idx]} (n={num_t})')
        filename = f'PSD_{opt_label}profile_G_{mouse_group_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)


def draw_psd_domain_power_timeseries_individual(psd_domain_power_timeseries_df, epoch_len_sec, y_label, output_dir, domain, opt_label=''):
    mouse_groups = psd_domain_power_timeseries_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]  

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
    y_max = np.nanmax(np.array([ts_stats[0] for ts_stats in domain_power_timeseries_stats_list])) * 1.1

    x_max = ts_mat.shape[0]
    x = np.arange(x_max)
    for i, profile in enumerate(hourly_ts_list):
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)
        sc.set_common_features_domain_power_timeseries(ax1, x_max, y_max)

        ax1.set_ylabel(y_label)
        ax1.set_xlabel('Time (hours)')

        ax1.plot(x, profile, color=stage.COLOR_NREM)

        fig.suptitle(
            f'Power timeseries: {"  ".join(psd_domain_power_timeseries_df.iloc[i,0:4].values)}')

        filename = f'power-timeseries_{domain}_{opt_label}I_{"_".join(psd_domain_power_timeseries_df.iloc[i,0:4].values)}'
        sc.savefig(output_dir, filename, fig)


def draw_psd_domain_power_timeseries_grouped(psd_domain_power_timeseries_df, epoch_len_sec, y_label, output_dir, domain, opt_label=''):
    mouse_groups = psd_domain_power_timeseries_df['Mouse group'].values
    mouse_groups_set = sorted(set(mouse_groups), key=list(
        mouse_groups).index)  # unique elements with preseved order
    bidx_group_list = [mouse_groups == g for g in mouse_groups_set]  

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
    y_max = np.nanmax(np.array([ts_stats[0] for ts_stats in domain_power_timeseries_stats_list])) * 1.1
    x = np.arange(x_max)
    if len(mouse_groups_set) > 1:
        # contrast to group index = 0
        for g_idx in range(1, len(mouse_groups_set)):
            num = np.sum(bidx_group_list[g_idx])
            fig = Figure(figsize=(13, 6))
            ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)

            sc.set_common_features_domain_power_timeseries(ax1, x_max, y_max)

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
                f'Power timeseries: {mouse_groups_set[0]} (n={num_c}) v.s. {mouse_groups_set[g_idx]} (n={num})')
            filename = f'power-timeseries_{domain}_{opt_label}G_{mouse_groups_set[0]}_vs_{mouse_groups_set[g_idx]}'
            sc.savefig(output_dir, filename, fig)
    else:
        # single group
        g_idx = 0
        num = np.sum(bidx_group_list[g_idx])
        fig = Figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111, xmargin=0, ymargin=0)

        sc.set_common_features_domain_power_timeseries(ax1, x_max, y_max)

        y = domain_power_timeseries_stats_list[g_idx][0, :]
        y_sem = domain_power_timeseries_stats_list[g_idx][1, :]/np.sqrt(num)
        ax1.plot(x, y, color=stage.COLOR_NREM)
        ax1.fill_between(x, y - y_sem,
                        y + y_sem, color=stage.COLOR_NREM, alpha=0.3)
        ax1.set_ylabel(y_label)
        ax1.set_xlabel('Time (hours)')

        fig.suptitle(f'Power timeseries: {mouse_groups_set[g_idx]} (n={num})')
        filename = f'power-timeseries_{domain}_G_{opt_label}{mouse_groups_set[g_idx]}'
        sc.savefig(output_dir, filename, fig)