@echo off
chcp 65001

set PYTHON=D:\Study\codeWork\python\faster2\.venv\Scripts\python.exe


set DATA_DIR=G:\EEG\experiment-data\FASTER_20191101_EEG_2019-013\data\
set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-013\result\
%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
%PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4