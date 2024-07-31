
import pandas as pd
import os
import numpy as np
from glob import glob
import locale
import chardet


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