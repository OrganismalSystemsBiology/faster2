## What is it?
FASTER2 is a set of Python scripts that automatically identify the sleep stages of mice by analyzing EEG/EMG signals. There is also a simple viewer of the analyzed data so that a user can visually inspect the results of FASTER2.

FASTER2 succeeds many good concepts but is written from scratch and uses different algorithms from its predecessor, FASTER ([Sunagawa et al. in 2013](https://onlinelibrary.wiley.com/doi/abs/10.1111/gtc.12053)).

## Main features
- Automatic sleep staging (REM, NREM, and Wake) of mouse EEG/EMG data.
- Basis statistical analyses based on the staging.
- There is a simple viewer to support human inspection of the staging, [signal_view](https://github.com/lsb-riken/signal_view).

## Where to download it

You can download release versions at:
https://github.com/lsb-riken/faster2/releases

Or the current version is available at:
https://github.com/lsb-riken/faster2/

## Installation
You need to have [Python](https://www.python.org/) to run FASATER2. Because FASTER2 depends on several python libraries, run the following commands to prepare FASTER2's virtual environment. We are developing and testing FASTER2 on Windows11 (x64), but it probably works on other platforms such as Linux or macOS.

### Python for Windows
Run an installer available at Python's official website:
https://www.python.org/downloads/windows/

Note: Check the 'Add Python to PATH' checkbox at the first step in the installer.

At the time of writing this document (2022/07/21), it is recommendable to use python 3.10.x.


In the windows shell, execute the following commands:
```sh
C:\Users\user-name> mkdir faster2-env
C:\Users\user-name> cd faster2-env
C:\Users\user-name> py -3.10 -m venv .venv  # replace "-3.10" with the version of Python you downloaded

C:\Users\user-name> .venv\Scripts\activate # You enter the virtual environment with this command
```
In the virtual environment, install the libraries: 

```sh
pip install -r requirements.txt # requirements.txt is in the downloaded FASTER2
or
pip install pandas hmmlearn matplotlib dask chardet mne==0.19.2 python-pptx
```

## How to use

There are three steps to RUN a session of FASTER2 analysis: 
- Prepare a working directory for the session that contains FASTER2 scripts, EEG/EMG data, and experiment information.
- Copy the recorded EEG/EMG data and make CSV files describing the experiment information (recording date, mouse ID, etc.).
- Run the FASTER2 scripts.

We recommend you have a "working directory" for each session of FASTER2 so that the directory keeps the related information in one place. The following sections describe each of the three steps in more detail.

### Prepare a working directory
1. Copy the downloaded FASTER2 files and data/ directory into, for example, [a big drive]/FASTER2_Rec001.
1. Open the sample_run.bat file with an editor, and edit the path to your Python executable.

Edit your sample_run.bat:
```
set PYTHON=[path to your python executable]
```
to something like: 
```
set PYTHON=C:\Users\user-name\faster2-env\.venv\Scripts\python.exe
```

You may want to reuse this directory as your "template" by saving it as, for example, "FASTER2_template". In future analysis, you can start by copying this template directory.

### Copy the recorded EEG/EMG data and write the experiment information
Copy your raw data into the [a big drive]/FASTER2_Rec001/data directory.
   - If your data is from telemetry devices of [DSI; Data Science International](https://www.datasci.com/), you need to export the EEG/EMG data into CSV files and put them in the data/DSI.txt directory.
   - If your data is an EDF file, just put the EDF file in the data/ directory.

The two CSV files in the data directory describe the information necessary for FASTER2 to analyze the EEG/EMG data.

#### exp.info.csv
This file describes parameters common to all the records in the EEG/EMG data:
|Experiment label|Rack label|Start datetime|End datetime|Sampling freq|
|----            |----      |----          |----        |----         |
|EEG_2020-001    |EEG_A-E   |2020/05/25 08:00:00|2020/05/27 08:00:00|100|

Since the downloaded FASTER2 directory has an example exp.info.csv file, you can simply edit it to describe your experiment. Be aware; that you must keep the header line unchanged because FASTER2 uses the headers to parse the CSV file.

#### mouse.info.csv
This file describes information about the individual records in the EEG/EMG data:
|Device label|Mouse group|Mouse ID|DOB|Stats report| Note|
|----   |----|----|----|----|----|
|ch01|WT|ES015-5-G7_1  |2019/03/25| Yes| Left 1(B1) |
|ch02|WT|ES015-5-G7_2  |2019/03/25| Yes| Left 2(B2) |
|ch03|WT|ES015-5-G7_3  |2019/03/25| No | Left 3(B3) cable bitten|
|ch04|WT|ES015-5-G7_4  |2019/03/25| Yes| Left 4(B4) | 
|ch05|MT|ES020-1-1-G6_1|2019/03/24| Yes| Right 1(C1)|
|ch06|MT|ES020-1-1-G6_2|2019/03/24| Yes| Right 2(C2)|
|ch07|MT|ES020-1-1-G6_3|2019/03/24| Yes| Right 3(C3)|
|ch08|MT|ES020-1-1-G6_4|2019/03/24| Yes| Right 4(C4)|

* Device label: An identifier of the recording device. This label must be the same as the label used for each record in the data files.
* Mouse group: The plot_summary.py calculates statistics according to this group. 
* Mouse ID: Label of the individual animal.
* DOB: Date of birth of the animal. At present, Faster2 does not use this information. But it is always a good practice to record DOB as age affects sleep time.
* Stats report: "Yes" or "No." If it is No, that record will be excluded from the statistics.
* Note: Additional information.

_Note_ The **control** group of mice should be placed first on the list. The summary.py assumes the first group of the list to be the control group and performs the statistical analyses accordingly.


### Run FASTER2 script

Then, run the sample_run.bat. The bat file executes four Python scripts. The first two scripts perform the main FASATER2 analysis (i.e., staging and calculating the basic statistics). These main processes usually take just a couple of minutes per animal. The latter scripts plot many graphs of voltage time-series and spectrums. These scripts are optional but useful for human visual inspection. The plotting takes about two hours for 8 animals x 5 days of recordings on a PC of moderate specs in 2022.


### Video
As it is not feasible to have a huge video file with a length of days, there are usually a set of multiple video clips (from minutes to hours) for each animal. These video files need to be arranged in a folder of the corresponding animal. Our viewer [signal_view](https://github.com/lsb-riken/signal_view) assumes that the set of folders is placed in the "video" folder in the FASTER2 folder. Also, the names of individual folders must be the same as the device labels in mouse.info.csv.

_note_ Each video file must contain a data time in its file name to indicate when the file started recording. For example, the filename should be like: CAM-61E0-C1_2018-01-27_06-58-47.avi. 


```
-+ FASTER2_experimentID-000/
 |
 +- data/
 +- result/
 +- summary/
 +- video/
```

There are three scripts to process videos.
* video_convert.py
* video_make_info.py
* video_split.py 

Because [signal_view](https://github.com/lsb-riken/signal_view) can recognize only a couple of video codecs, you probably need to convert video files by using video_convert.py. Because convert_video.py is just a utility script to call [ffmpeg](https://ffmpeg.org/), you need to install it before using this script.


```sh
 python video_convert.py -t g:\tmp_video -o g:\FASTER2_Rec001\video -w 2 -e h264_nvenc
 ```
 In the example above, I specified the encoder option (-e) as h264_nvenc to utilize my NVIDIA GPU. The default is libx264 which uses only CPU. Also, you may want to search for an optimal worker number (-w option that specifies the number of processes that run in parallel) for your PC.

 [signal_view](https://github.com/lsb-riken/signal_view) needs to know when each video file started recording and how long it was. The information is stored in video.info.csv for each mouse. You can generate the video_info.csv files of mice in the video/ folder by using video_make_info.py.

 ```sh
 python make_video_info.py -t FASTER2_experimentID-000/video
 ```
 
 video_split.py is just a utility script to make a short video clip. This may be useful when you find an interesting behavior of an animal and want to make a short video clip for a presentation.
