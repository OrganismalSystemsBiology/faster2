@echo off
chcp 65001

set PYTHON=[path to your python executable]
set FASTER2_DIR="."

set DATA_DIR=%FASTER2_DIR%\data\
set RESULT_DIR=%FASTER2_DIR%\result\

%PYTHON% stage.py -d %DATA_DIR% -r %RESULT_DIR% -p
%PYTHON% summary.py -f %FASTER2_DIR% 
%PYTHON% slide.py -s %FASTER2_DIR%\summary 

%PYTHON% plot_hypnogram.py -d %DATA_DIR% -r %RESULT_DIR%
%PYTHON% plot_timeseries.py -d %DATA_DIR% -r %RESULT_DIR% -w 4
%PYTHON% plot_spectrum.py -d %DATA_DIR% -r %RESULT_DIR% -w 4