## What is it?
FASTER2 is a set of Python scripts for automatically identifying sleep stages of mouse by analyzing EEG/EMG signals. There is also a simple viewer of the analyzed data so that a user can visually inspect the results of FASTER2.

FASTER2 is a successor of FASTER [Sunagawa et al. in 2013](https://onlinelibrary.wiley.com/doi/abs/10.1111/gtc.12053). FASTER2 is written from scratch and uses different algorithms from the predecessor.

## Main features
- Automatic sleep staging (REM, NREM, and Wake) of mouse EEG/EMG data.
- A simple viewer to support human inspection of the staging 

## Where to download it

You can download release versions at:
https://github.com/ygriku/faster2/releases

Or the current version are available at:
```sh
git clone git@github.com:ygriku/faster2.git # via SSH
or
git clone https://github.com/ygriku/faster2.git # via HTTPS
```

## Installation
You need to have [Python](https://www.python.org/) to run FASATER2. It is recommendable to have a virtual environment for FASTER2. Because FASTER2 depends on several python libraries, run the following command in the FASTER2's virtual environment. We are developing and testing FASTER2 on Windows10 (x64), but probably it works on other platforms such as Linux or macOS.

### Windows10
Run an installer available at the python's official web site:
https://www.python.org/downloads/windows/

Note: Check the 'Add python to PATH' checkbox at the first step in the installer.

At the time of writing this document (2020/05/26), it is recommendable to use python 3.7.x.


In the windows's shell, execute the following commands:
```sh
C:\Users\user-name> mkdir faster2-env
C:\Users\user-name> cd faster2-env
C:\Users\user-name> py -3.7 -m venv .venv  # replace "-3.7" with the version of Python you downloaded

C:\Users\user-name> .venv\Scripts\activate # You enter the virtual environment with this line
```
In the virtual environment, install the libraries: 

```sh
pip install -r requirements.txt # requirements.txt is in the downloaded FASTER2
or
pip install pandas hmmlearn matplotlib pillow dask[complete] chardet mne==0.19.2
```

## How to use

There are three steps to RUN a FASTER2 analysis. We recommend you to have a "working directory" for each session of FASTER2 so that the directory keeps the related information in one place.

- Prepare a working directory for FASTER2 scripts, EEG/EMG data, and experiment information.
- Copy the data and write the experiment information (recording date, mouse ID etc.) into CSV files.
- Run the FASTER2 script.

### Prepare a working directory
1. Copy the downloaded FASTER2 files and data/ directory to, for example, [a big drive]/FASTER2_Rec001.
1. Open the sample_run.bat file with an editor, and edit the path to your Python executable.

Edit your sample_run.bat:
```
set PYTHON=[path to your python executable]
```
to something like: 
```
set PYTHON=C:\Users\user-name\faster2-env\.venv\Scripts\python.exe
```

You can use this modified delectory as your "template" directory by savint it as, for example, "FASTER2_template". In the future analysis, you can start from copying & renaming the directory instead of starting from the downloaded directory.

### Copy the data and write the experiment information
Copy your raw data into the [a big drive]/FASTER2_Rec001/data directory.
   - If your data is from telemetry devices of [DSI; Data Science International](https://www.datasci.com/), you need to export EEG/EMG data into a CSV file:
   - If your data is an EDF file, just put the EDF file in the data/ directory.

The two CSV files in the directory describe the experiment information necessary to
perform FASTER2 analysis.

#### exp.info.csv
This file describes experiment parameters common to all the recorded mice:
|Experiment label|Rack label|Start datetime|End datetime|Sampling freq|
|----            |----      |----          |----        |----         |
|EEG_2020-001    |EEG_A-E   |2020/05/25 08:00:00|2020/05/27 08:00:00|100|

Since the downloaded FASTER2 folder has an example exp.info.csv file, you can simply change it to describe your experiment. Be aware; you must keep the header line unchanged because FASTER2 uses the headers to parse the CSV file.

#### mouse.info.csv
This file describes information about the individual mouse in the exeriment:
|Device label|Mouse group|Mouse ID|DOB|Stats report| Note|
|----   |----|----|----|----|----|
|ID47395|WT|ES015-5-G7_1  |2019/03/25| Yes| Left 1(B1) |
|ID58703|WT|ES015-5-G7_2  |2019/03/25| Yes| Left 2(B2) |
|ID47479|WT|ES015-5-G7_3  |2019/03/25| No | Left 3(B3) cable bitten|
|ID47313|WT|ES015-5-G7_4  |2019/03/25| Yes| Left 4(B4) | 
|ID47481|MT|ES020-1-1-G6_1|2019/03/24| Yes| Right 1(C1)|
|ID45791|MT|ES020-1-1-G6_2|2019/03/24| Yes| Right 2(C2)|
|ID45764|MT|ES020-1-1-G6_3|2019/03/24| Yes| Right 3(C3)|
|ID46770|MT|ES020-1-1-G6_4|2019/03/24| Yes| Right 4(C4)|

* Device label: An identifier of the recording device. This must be same with the labels used in the data files.
* Mouse group: The plot_summary.py calculates statistics (e.g. means) of mice in the same group. 
* Mouse ID: Label of the individual mice.
* DOB: Date of birth of the mouse
* Stats report: "Yes" or "No". If it is Yes, the it will be included in the statistics.
* Note: Additional information.

### Run FASTER2 script

Then, run the sample_run.bat. The bat file executes four Python scripts. The first two scripts perform the main FASATER2 analysis (staging and basic summary statistics). This main process takes about a couple of minutes per mouse, depending on the input data format and size. The latter two scripts plot many graphs of voltage time-series and spectrums. These latter scripts are optional but useful for human visual inspection. The plotting process takes about 60 minutes for 8 mice x 2 days recordings on a PC of moderate specs in the middle of 2020.

