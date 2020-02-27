@echo off
chcp 65001

set PYTHON=D:\Study\codeWork\python\faster2\.venv\Scripts\python.exe
rem set DATA_DIR=G:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\data\
rem set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\result\

rem set DATA_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-015\data\
rem set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-015\result\

set DATA_DIR=G:\EEG\experiment-data\FASTER_20191108_EEG_2019-016\data\
set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191108_EEG_2019-016\result\

rem set DATA_DIR=\\Sss-analysis\faster_20181023\FASTER_20200213_EEG_2019-024\data
rem set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200213_EEG_2019-024\result\

rem set DATA_DIR=\\Sss-analysis\faster_20181023\FASTER_20200128_EEG_2019-022\data
rem set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200128_EEG_2019-022\results\

%PYTHON% stage.py --data_dir %DATA_DIR% --result_dir %RESULT_DIR% -p


