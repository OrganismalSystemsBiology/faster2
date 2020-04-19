@echo off
chcp 65001

set PYTHON=D:\Study\codeWork\python\faster2\.venv\Scripts\python.exe

set FASTER2_DIR=G:\EEG\experiment-data\FASTER_20191101_EEG_2019-013

set DATA_DIR=%FASTER2_DIR%\data\
set RESULT_DIR=%FASTER2_DIR%\result\
%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
%PYTHON% plot_summary.py -d %FASTER2_DIR% -o summary 
%PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4