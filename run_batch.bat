@echo off
chcp 65001

set PYTHON=D:\Study\codeWork\python\faster2\.venv\Scripts\python.exe


REM set DATA_DIR=G:\EEG\experiment-data\FASTER_20191101_EEG_2019-013\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-013\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM set DATA_DIR=G:\EEG\experiment-data\FASTER_20191108_EEG_2019-014\data
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191108_EEG_2019-014\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


set DATA_DIR=G:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\data\
set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200206_EEG_2019-023\result\
%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
%PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4

set DATA_DIR=\\Sss-analysis\faster_20181023\FASTER_20200213_EEG_2019-024\data
set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200213_EEG_2019-024\result\
%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
%PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4

REM set DATA_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-015\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191101_EEG_2019-015\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM set DATA_DIR=G:\EEG\experiment-data\FASTER_20191108_EEG_2019-016\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191108_EEG_2019-016\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM set DATA_DIR=G:\EEG\experiment-data\FASTER_20191129_EEG_2019-017\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20191129_EEG_2019-017\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM set DATA_DIR=\\Sss-analysis\faster_20181023\FASTER_20200128_EEG_2019-022\data
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER2_20200128_EEG_2019-022\results\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM set DATA_DIR=G:\EEG\experiment-data\FASTER_170331_Wake36A\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER_170331_Wake36A\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM set DATA_DIR=G:\EEG\experiment-data\FASTER_170406_Wake37A\data\
REM set RESULT_DIR=G:\EEG\experiment-data\FASTER_170406_Wake37A\result\
REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p


REM %PYTHON% stage.py %DATA_DIR% %RESULT_DIR% -p
rem %PYTHON% plot_timeseries.py %DATA_DIR% %RESULT_DIR%