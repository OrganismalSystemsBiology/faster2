
import pandas as pd
import os
import io
import json
import re
import pyedflib
import numpy as np
from glob import glob
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath, PurePosixPath
import locale
import chardet
import pickle
import faster2lib.dsi_tools as dt

def print_log(msg):
    if 'log' in globals():
        log.debug(msg)
    else:
        print(msg)

def read_stages(stage_folder_path, label, type='auto'):
    stage_file_path = os.path.join(stage_folder_path, f'{label}.{type}.stage.csv')

    # read stages
    stage_file = glob(stage_file_path)
    if len(stage_file) != 1:
        if len(stage_file) == 0: 
            raise LookupError(f'no stage file found:{stage_file_path}')
        elif len(stage_file)>1:
            raise LookupError(f'too many stage files found:{stage_file}')

    stage_file = stage_file[0]
    stages = pd.read_csv(stage_file, engine='python', skiprows=7, header=None)
    regularized_stages = np.array([st.strip().upper() for st in stages.iloc[:,0].values])

    bidx = ((regularized_stages == 'NREM') | (regularized_stages == 'REM') | (
        regularized_stages == 'WAKE') | (regularized_stages == 'UNKNOWN'))

    if len(regularized_stages) != np.sum(bidx):
        raise ValueError(f'Uninterpretable annotation was found at lines {np.where(~bidx)} in the stage file:{stage_file}')
    
    return regularized_stages


def read_stages_with_eeg_diagnosis(stage_folder_path, label, type='auto'):
    stage_file_path = os.path.join(stage_folder_path, f'{label}.{type}.stage.csv')

    # read stages
    stage_file = glob(stage_file_path)
    if len(stage_file) != 1:
        if len(stage_file) == 0: 
            raise LookupError(f'no stage file found:{stage_file_path}')
        elif len(stage_file)>1:
            raise LookupError(f'too many stage files found:{stage_file}')

    stage_file = stage_file[0]
    stages = pd.read_csv(stage_file, engine='python', skiprows=7, header=None)
    regularized_stages = np.array([st.strip().upper() for st in stages.iloc[:,0].values])
    nan_eeg = stages.iloc[:, 4].to_numpy() # NaN ratio EEG-TS
    outlier_eeg = stages.iloc[:, 6].to_numpy() # Outlier ratio EEG-TS
    
    bidx = ((regularized_stages == 'NREM') | (regularized_stages == 'REM') | (
        regularized_stages == 'WAKE') | (regularized_stages == 'UNKNOWN'))

    if len(regularized_stages) != np.sum(bidx):
        raise ValueError(f'Uninterpretable annotation was found at lines {np.where(~bidx)} in the stage file:{stage_file}')
    
    return regularized_stages, nan_eeg, outlier_eeg


def patch_nan(y):
    """ patches nan elements by non-nan elements in the top of itself.
    This function leaves nan if more than a half of the given array is
    nan.

    Notice: this function modifies the given array itself.

    
    Args:
        y (np.array(1)): an array to be patched

    Returns:
        float: a ratio of nan
    """
    bidx_nan = np.isnan(y)
    nan_len = np.sum(bidx_nan)
    
    # y is pachable only if nan_len < len(y)/2
    if nan_len <= len(y)/2:
        # the y is patched by non-nan elements in the top of itself
        y[bidx_nan] = y[~bidx_nan][0:nan_len]
    
    return nan_len/len(y)


def encode_lookup(target_path):
    """
    Tries to find what encoding the target file uses.

    input
        target_path

    output
        encoding string
    """
    code_list = ['cp932','euc_jisx0213', 'iso2022jp', 'iso2022_kr','big5','big5hkscs','johab','euc_kr','utf_16','iso8859_15','latin_1','ascii', 'Unknown']

    # First ask chardet
    readinSize = 1024*2000 # readin 2MB for checking the encoding
    with open(target_path, 'rb') as fh:
        data = fh.read(readinSize)
    enc = chardet.detect(data)['encoding']

    # chardet and OS's preferred encoding is the first candidates
    code_list = [enc, locale.getpreferredencoding()] + code_list

    # Find the encoding that can read the target_file without Exception
    for enc in code_list:
        try:
            with open(target_path,  "r", encoding=enc) as f:
                f.readlines()
            break
        except:
            pass

    # fallback to the OS default in case all attempts failed 
    if enc=='Unknown':
        enc = locale.getpreferredencoding()

    return enc


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
        codename = encode_lookup(filepath)
    except LookupError as lkup_err:
        print_log(lkup_err)
        exit(1)

    print_log(f'Reading mouse info file:{filepath}')
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


def load_collected_mouse_info(summary_dir):
    """ 
    read the collected mouse information used for the summary
    Here is the 'epoch range' information
    """
    def _str_to_datetime(s):
        try:
            dt_str = datetime.strptime(s, format_str)
        except ValueError as err_v:
            if len(err_v.args) > 0 and err_v.args[0].startswith('unconverted data remains: '):
                s = s[:-1] # 'remove the last character 'Z' for backward compatibility
                dt_str = datetime.strptime(s, format_str)       
        return dt_str

    try:
        collected_mouse_info_path = os.path.join(summary_dir, 'collected_mouse_info_df.json')
        print_log(f'Reading collected_mouse_info:\n'
                  f'    {collected_mouse_info_path}')
        with open(collected_mouse_info_path, 'r',
                    encoding='UTF-8') as infile:
            mouse_info_collected = json.load(infile)
    except FileNotFoundError as err:
        print_log(
            f'Failed to find collected_mouse_info_df.json. Check the summary folder path is valid.\n'
            f'{err}')
        exit(1)
    
    # This is a patch for bridging the inconsistency between Pandas 1.5 and 2.0
    json_str = mouse_info_collected['mouse_info']
    json_str_wrapped = io.StringIO(json_str.replace("datetime", "string"))

    mouse_info_collected['mouse_info'] = pd.read_json(json_str_wrapped, orient="table")

    format_str = "%Y-%m-%dT%H:%M:%S.%f"
    mouse_info_collected['mouse_info']['exp_start_string'] = [_str_to_datetime(a) for a in mouse_info_collected['mouse_info']['exp_start_string']]

    return mouse_info_collected


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

        signal_pre, signal_header_pre, header_pre = pyedflib.highlevel.read_edf(edf_file, ch_names=[f'EEG{device_id}', f'EMG{device_id}'])
        if signal_pre.shape[0] < 2:
            raise ValueError('Failed to load EEG/EMG signals. Check the channel names in the EDF file are consistent with mouse.info.csv.')

        measurement_start_datetime = header_pre['startdate']
        if isinstance(start_datetime, datetime):
            start_offset_sec = (
                start_datetime - measurement_start_datetime).total_seconds()
            end_offset_sec = start_offset_sec + epoch_num * epoch_len_sec
            
            signal_range = slice(int(start_offset_sec*sample_freq), int(end_offset_sec*sample_freq))
            eeg = signal_pre[0][signal_range]
            emg = signal_pre[1][signal_range]
        else:
            eeg = signal_pre[0]
            emg = signal_pre[1]

        eeg_vm = eeg.reshape(-1, epoch_len_sec * sample_freq)
        emg_vm = emg.reshape(-1, epoch_len_sec * sample_freq)
        epoch_num_read = eeg_vm.shape[0]
        if epoch_num_read != epoch_num:
            raise ValueError(f'Number of read epochs;{epoch_num_read} is inconsistent with exp.info.csv;{epoch_num}.\n'
                            f'Check the specified start_datetime:{start_datetime}, recording start_datetime:{measurement_start_datetime},'
                            f' or sample frequency:{sample_freq} is correct\n')
    elif os.path.exists(os.path.join(data_dir, 'dsi.txt')):
        # try to read dsi.txt
        not_yet_pickled = True
        try:
            dsi_reader_eeg = dt.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'),
                                               f'{device_id}', 'EEG',
                                               sample_freq=sample_freq)
            dsi_reader_emg = dt.DSI_TXT_Reader(os.path.join(data_dir, 'dsi.txt/'),
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


class DataSet:
    """ Class to manage the dataset used for the previous analysis (stage.py or summary.py)
    """
    def __init__(self, mouse_info_collected_df, sample_freq, epoch_len_sec, 
                 epoch_num, epoch_range_target, basal_days, stage_ext, new_root_dir=None, path_style='windows'):
        """ Initialize the dataset
        Args:
            mouse_info_collected_df (DataFrame): a DataFrame that contains the collected information of the mice
            sample_freq (int): sampling frequency
            epoch_len_sec (int): the length of an epoch in seconds
            epoch_num (int): the number of all epochs processed in the summary analysis
            epoch_range_target (slice): an epoch range targeted for the analysis with this dataset
            stage_ext (str): a stage file extension
            basal_days (int): the number of days used as the basal period for TDD scaling
            new_root_dir (str): a new root directory of the data
            path_style (str): the style of the path used in the mouse info ('windows' or 'posix')
        
        """
        self.mouse_info_collected_df = mouse_info_collected_df.copy()
        self.sample_freq = sample_freq
        self.epoch_len_sec = epoch_len_sec
        self.epoch_num = epoch_num
        self.epoch_range_target = epoch_range_target
        self.basal_days = basal_days
        self.stage_ext = stage_ext
        self.path_style = path_style

        # change the root directory of the data if needed
        if new_root_dir:
            for i, r in self.mouse_info_collected_df.iterrows():
                if self.path_style == 'windows':
                    faster_dir = PureWindowsPath(r['FASTER_DIR'])
                elif self.path_style == 'posix':
                    faster_dir = PurePosixPath(r['FASTER_DIR'])
                else:
                    print_log(
                        f'Failed to set the path style to: \"{path_style}\" trying to use \"windows\".')
                    faster_dir = PureWindowsPath(r['FASTER_DIR'])
                new_faster_dir = Path(new_root_dir, *faster_dir.parts[1:]) # Path (windows or posix) can change depending on the current system
                self.mouse_info_collected_df.loc[i, 'FASTER_DIR'] = new_faster_dir

        # set the default stage_ext
        if not self.stage_ext:
            self.stage_ext = 'faster2'

    def load_vm(self, idx: int):
        """ Load the voltage matrix of the mouse specified by the index

        Args:idx    index of the mouse_info_collected_df
        
        """
        r = self.mouse_info_collected_df.iloc[idx]
        device_label = r['Device label'].strip()
        faster_dir = Path(r['FASTER_DIR'])
        data_dir = faster_dir / 'data'
        start_date_time = r['exp_start_string']

        eeg_vm, emg_vm, _ = read_voltage_matrices(data_dir, device_label, self.sample_freq, 
                                 self.epoch_len_sec, self.epoch_num, start_date_time)
        
        if self.epoch_range_target:
            eeg_vm = eeg_vm[self.epoch_range_target,:]
            emg_vm = emg_vm[self.epoch_range_target,:]

        return eeg_vm, emg_vm

    
    def load_stage(self, idx: int):
        """ Load the data of the mouse specified by the index

        Args:idx    index of the mouse_info_collected_df
        
        """
        r = self.mouse_info_collected_df.iloc[idx]
        device_label = r['Device label'].strip()
        faster_dir = Path(r['FASTER_DIR'])
        result_dir = faster_dir / 'result'
        
        stages = read_stages(result_dir, device_label, self.stage_ext)    

        if self.epoch_range_target:
            stages = stages[self.epoch_range_target]

        return stages