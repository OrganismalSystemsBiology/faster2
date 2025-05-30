import faster2lib.eeg_tools as et
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from glob import glob
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

class DSI_TXT_Reader:
    def __init__(self, dataroot, label, signal_type, epoch_len_sec=8, sample_freq=100):
        # arguments 
        #   dataroot ... path to folder containing the datafiles
        #   label ...  e.g. ID46553
        #   datatype ... 'EEG' or 'EMG'
        #   sampling_freq ... sampling frequency (default 100 Hz)

        self._dataroot = dataroot
        self._label = label
        self._filepaths_of_opened_file = []
        self._data_buffer = pd.DataFrame()
        self._signal_type = signal_type
        self._epoch_len_sec = epoch_len_sec
        self._sample_freq = sample_freq
        self.norm_fac = 1
        
    # read epochs using the epoch index in the directory
    # arguments 
    #   signal_type ... EEG or EMG
    #   epoch_start ... starting epoch number (1-start index, inclusive)
    #   epoch_end ... ending epoch number (1-start index, inclusive)
    def read_epochs(self, epoch_start, epoch_end):
        idx_start = int((epoch_start - 1)*self._epoch_len_sec*self._sample_freq) # 0-start index
        idx_end   = int(epoch_end*self._epoch_len_sec*self._sample_freq - 1) # 0-start index

        hour_start = int(idx_start/self._sample_freq/3600)
        hour_end   = int(idx_end/self._sample_freq/3600)
        
        filepaths = []
        for hour in range(hour_start, hour_end+1):
            file_number = hour
            filename = '%s.%s.%04d.txt' % (self._label, self._signal_type, file_number)
            filepaths.append(os.path.join(self._dataroot, filename))

        # check encodings
        enc = et.encode_lookup(filepaths[0])

        if not set(filepaths).issubset(self._filepaths_of_opened_file):
            # read files
            dask_dataframe = dd.read_csv(filepaths,
                                         engine='python',
                                         dtype={'time':np.float64, 'value':np.float64}, 
                                         names=['time', 'value'], skiprows=5, header=None,
                                         encoding=enc)
            print(f'Reading {len(filepaths)} {self._signal_type} files in {self._dataroot}:')
            with ProgressBar():
                self._data_buffer = dask_dataframe.compute()
            
            # update filepahts list
            self._filepaths_of_opened_file = filepaths
        

        sec_start = (epoch_start - 1)*self._epoch_len_sec
        sec_end   = epoch_end*self._epoch_len_sec
        
        idx_start_in_buf = np.mod(idx_start, 3600*self._sample_freq)
        idx_end_in_buf = 3600*self._sample_freq*hour_end + np.mod(idx_end, 3600*self._sample_freq)
        buf_values = self._data_buffer.value.values
        values = np.copy(buf_values[idx_start_in_buf:(idx_end_in_buf+1)])
        times = np.arange(sec_start, sec_end, 1/self._sample_freq)
        
        res_data = pd.DataFrame({'time':times, 'value':values})
        
        return(res_data)


    # read epochs using the datetime information in the comment line
    # arguments 
    #   signal_type ... EEG or EMG
    #   start_datetime ... starting datetime (inclusive)
    #   end_datetime   ... ending datetime (not inclusive)
    def read_epochs_by_datetime(self, start_datetime, end_datetime):

        glob_path = os.path.join(self._dataroot, f'{self._label}.{self._signal_type}*')
        file_list = glob(glob_path)
        if len(file_list) == 0: 
            raise LookupError(f'No dsi.txt file found:{glob_path}')


        # check encodings
        enc = et.encode_lookup(file_list[0])

        # get start datetimes of each file
        print('Parsing the start time of each file')
        file_datetimes = []
        pat = re.compile(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})')
        for path in file_list:
            with open(path, 'r', encoding=enc) as f:
                f.readline() # ignore the first line
                second_line = f.readline() # take the second line
            
            # Assuming the second line contains a line like '# Time: 2018/01/26 11:56:00'
            pat = re.compile(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})')
            mat = pat.search(second_line)    
            
            if mat:
                datetime_str = mat.group(1)
            else:
                print(f'Failed to get start datetime of {path}')
                raise(AttributeError(f'Failed to get start datetime of {path}'))
            
            file_start_datetime = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')   
            file_datetimes.append(file_start_datetime)
        file_datetimes = np.array(file_datetimes)

        # make the binary index of the interval between start_ and end_datetime
        bidx_targetfiles = (file_datetimes >= start_datetime) & (file_datetimes < end_datetime)
        # include the previous file of the first file if the first file does not start at the start_datetime
        idx_first = np.where(bidx_targetfiles)[0][0] # first file's index
        if idx_first > 0 and file_datetimes[idx_first] > start_datetime:
            idx_first = idx_first - 1
            bidx_targetfiles[idx_first] = True

        filepaths = np.array(file_list)[bidx_targetfiles].tolist()

        if not set(filepaths).issubset(self._filepaths_of_opened_file):
            # read files
            dask_dataframe = dd.read_csv(filepaths,
                                         engine='python',
                                         dtype={'time':np.float64, 'value':np.float64}, 
                                         names=['time', 'value'], skiprows=5, header=None,
                                         encoding=enc)
            print(f'Reading {len(filepaths)} {self._signal_type} files in {self._dataroot}:')
            with ProgressBar():
                self._data_buffer = dask_dataframe.compute()
            
            # update filepahts list
            self._filepaths_of_opened_file = filepaths
        
            # assign absolute timestamps to rows
            print('Assigning absolute timestamps to rows')
            nrow = self._data_buffer.shape[0]
            times = np.arange(0, nrow)/self._sample_freq
            timestamps = [file_datetimes[idx_first] + timedelta(seconds=t) for t in times]
            self._data_buffer.time = timestamps

        bidx_target_rows =  (self._data_buffer.time >= start_datetime) & (self._data_buffer.time < end_datetime)
        buf_values = self._data_buffer.value.values
        values = np.copy(buf_values[bidx_target_rows])
        sec_start = (start_datetime - file_datetimes[0]).total_seconds()
        sec_end = (end_datetime - file_datetimes[0]).total_seconds()
        times = np.arange(sec_start, sec_end, 1/self._sample_freq)
        
        res_data = pd.DataFrame({'time':times, 'value':values})
        
        return(res_data)


