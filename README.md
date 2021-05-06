# iScore (under construction)
Rodent sleep scoring program

Current version has only manual scoring and scoring export/import enabled.
Score can be a python file (pkl), a matlab file form Matlab's iscore (.mat) or a text file with the epoch state in each line (.txt).

Scorings are saved as a pkl file and txt file, that can be loaded into Matlab's iScore.

# Installation
Download the files in this repo to a folder

Download and instal Python 3.8 (google it if you don't know how)

Open terminal and type:

python -m pip install --upgrade pip
pip install numpy
pip install scipy
pip install matplotlib
pip install pyqt5
pip install mne
pip install pyqtgraph

*Note:* On MAc/Linux type python3 and pip3 instead of python and pip.

# Execution

Open a terminal and type:

PC: python <repo path>\escore.py
  
Mac/Linux: python3 <repo path>/escore.py

![GUI](screenshotjpg.jpg)
