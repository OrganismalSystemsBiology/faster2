# -*- coding: utf-8 -*-
""" Useful functions for handling PSD
"""
from logging import getLogger
import os
import numpy as np
import pandas as pd
import faster2lib.eeg_tools as et
import stage

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

        psd_summary_df = psd_summary_df.append([
            [exp_label, mouse_group, mouse_id, device_label,
             'REM', np.sum(bidx_rem_target)] + psd_summary_rem.tolist()], ignore_index=True)
        psd_summary_df = psd_summary_df.append([
            [exp_label, mouse_group, mouse_id, device_label,
             'NREM', np.sum(bidx_nrem_target)] + psd_summary_nrem.tolist()], ignore_index=True)
        psd_summary_df = psd_summary_df.append([
            [exp_label, mouse_group, mouse_id, device_label,
             'Wake', np.sum(bidx_wake_target)] + psd_summary_wake.tolist()], ignore_index=True)

    freq_columns = [f'f@{x}' for x in freq_bins.tolist()]
    column_names = ['Experiment label', 'Mouse group',
                    'Mouse ID', 'Device label', 'Stage', 'epoch #'] + freq_columns
    psd_summary_df.columns = column_names

    return psd_summary_df


def make_target_psd_info(mouse_info_df, epoch_range, epoch_len_sec, sample_freq,
                         stage_ext, start_datetime):
    """makes PSD information sets for subsequent static analysis for each mouse:
    Arguments:
        mouse_info_df {pd.DataFram} -- a dataframe given by mouse_info_collected()
        sample_freq {int} -- sampling frequency
        epoch_range {slice} -- a range of target epochs
        stage_ext {str} -- a file sub-extention (e.g. 'faster2' for *.faster2.csv)
        start_datetime -- datetime object for reading the voltage matrices

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

        # good PSD should have the nan- and outlier-ratios of less than 1%
        bidx_good_psd = (nan_eeg < 0.01) & (outlier_eeg < 0.01)

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
