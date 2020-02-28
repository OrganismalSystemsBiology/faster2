@echo off
chcp 65001

set PYTHON=D:\Study\codeWork\python\faster2\.venv\Scripts\python.exe

REM set DATA_DIR=G:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\result\

REM set DATA_DIR=\\Sss-analysis\faster_20181023\FASTER_20200213_EEG_2019-024\data
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200213_EEG_2019-024\result\

REM set DATA_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-015\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-015\result\

REM set DATA_DIR=G:\EEG\experiment-data\FASTER_20191108_EEG_2019-016\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191108_EEG_2019-016\result\

set DATA_DIR=G:\EEG\experiment-data\FASTER_20191129_EEG_2019-017\data\
set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191129_EEG_2019-017\result\

REM set DATA_DIR=\\Sss-analysis\faster_20181023\FASTER_20200128_EEG_2019-022\data
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200128_EEG_2019-022\results\

REM set DATA_DIR=G:\EEG\experiment-data\FASTER_170331_Wake36A\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER_170331_Wake36A\result\

REM set DATA_DIR=G:\EEG\experiment-data\FASTER_170406_Wake37A\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER_170406_Wake37A\result\


%PYTHON% stage.py --data_dir %DATA_DIR% --result_dir %RESULT_DIR% -p