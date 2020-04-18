@echo off
chcp 65001

set PYTHON=C:\Users\rikuhiro\codework\python\faster2\.venv\Scripts\python.exe

set DATA_DIR=D:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\data\
set RESULT_DIR=D:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\result\
%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
REM %PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4

set DATA_DIR=D:\EEG\experiment-data\FASTER2_20200213_EEG_2019-024\data\
set RESULT_DIR=D:\EEG\experiment-data\FASTER2_20200213_EEG_2019-024\result\
%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
REM %PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4